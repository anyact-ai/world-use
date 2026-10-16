"""An agent-authored, measurement-driven packing procedure for this two-snack setup.

Initial image regions and gripper margins are task parameters. Object poses and
contact truth are never read here. Each invocation records fresh observations,
its source, fitted geometry, commands and camera-based placement checks.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from episode import LunchboxEpisode


def rectangle(points, *, height=None, prism=False):
    points = np.asarray(points)
    visible = points
    top = float(np.percentile(points[:, 2], 95)) if height is None else float(height)
    points = points[np.abs(points[:, 2]-top) < .0015]
    if len(points) < 40:
        raise ValueError('Not enough visible planar surface samples')
    xy = points[:, :2]
    origin = xy.mean(0)
    _, _, axes = np.linalg.svd(xy-origin, full_matrices=False)
    if axes[0, 0] < 0:
        axes[0] *= -1
    if np.linalg.det(axes) < 0:
        axes[1] *= -1
    # These packages are upright rectangular prisms: their visible side faces
    # also constrain the footprint when the hand occludes part of the top.
    extent_points = visible[:, :2] if prism else xy
    local = (extent_points-origin) @ axes.T
    lo, hi = np.percentile(local, [.5, 99.5], axis=0)
    center = origin+(lo+hi)/2 @ axes
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]], [hi[0], lo[1]]]) @ axes+origin
    return dict(center=center.tolist(), axes=axes.tolist(), size=(hi-lo).tolist(), top=top,
                source=np.column_stack([corners, np.full(4, top)]).tolist())


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T+pose[:3, 3]


class PackingTask:
    def __init__(self, episode):
        self.e = episode
        self.stages = {}
        self.report = dict(stages=self.stages, objects={}, checks={})

    def save(self):
        (self.e.out/'task.json').write_text(json.dumps(self.report, indent=2)+'\n')

    def call(self, stage, tool, intent, **kwargs):
        result = self.e.act(tool, intent=intent, **kwargs)
        self.stages[stage] = len(self.e.log)-1
        self.save()
        if result is None:
            raise RuntimeError(f'{stage}: tool failed')
        if tool == 'run' and result['status'] != 'done':
            raise RuntimeError(f'{stage}: {result}')
        if tool in ('select_target', 'measure_pixels') and not result['valid']:
            raise RuntimeError(f'{stage}: invalid measurement')
        return result

    def points(self, measurement):
        return np.load(self.e.out/'flight/perception'/f'{measurement["id"]}.npz')['points']

    def measure(self):
        self.call('card', 'card', 'Read robot limits and sensors')
        status = self.call('status', 'status', 'Confirm the simulated robot starts disabled')
        assert status['session']['mode'] == 'simulation' and not status['enabled']
        self.call('side', 'camera_frame', 'Introduce the fixed side camera', camera='side', depth=True)
        wrist = self.call('wrist', 'camera_frame', 'Inspect the snacks and smaller lunchbox',
                          camera='wrist', depth=True)
        self.initial = {}
        for name, box in [('blue', [765, 405, 1040, 580]), ('red', [800, 137, 950, 314]),
                          ('tray', [330, 100, 660, 560])]:
            self.initial[name] = self.call(name+'_initial', 'select_target', 'Select '+name+' in the wrist image',
                                          frame=wrist['frame'], target=name, box=box)
        floor = self.call('floor', 'measure_pixels', 'Measure a patch wholly inside the lunchbox floor',
                          frame=wrist['frame'], target='floor', box=[455, 250, 530, 390])
        counter = self.call('counter', 'measure_pixels', 'Measure the support beneath the snacks',
                            frame=wrist['frame'], target='counter', box=[690, 320, 750, 375])
        self.scene_evidence = [self.initial['tray'], floor, counter]
        self.floor = float(np.median(self.points(floor)[:, 2]))
        self.counter = float(np.median(self.points(counter)[:, 2]))
        self.tray = rectangle(self.points(self.initial['tray']), height=self.floor)
        self.fits = {n: rectangle(self.points(self.initial[n]), prism=True) for n in ('blue', 'red')}
        self.report.update(floor_m=self.floor, counter_m=self.counter, tray=self.tray,
                           fits=self.fits, floor_evidence=floor['id'], counter_evidence=counter['id'])
        # Center a compact pair in the measured usable floor. Preserve 10 mm
        # between visible surfaces plus a 2 mm shape allowance on each item.
        gap, allowance = .010, .002
        blue_w = self.fits['blue']['size'][1]+2*allowance
        red_w = self.fits['red']['size'][1]+2*allowance
        span = blue_w+red_w+gap
        tray_size = np.asarray(self.tray['size'])
        length = max(self.fits[n]['size'][0]+2*allowance for n in self.fits)
        if length+.020 > tray_size[0] or span+.014 > tray_size[1]:
            raise ValueError('Measured lunchbox is too small for the selected clearance margins')
        axes = np.asarray(self.tray['axes'])
        center = np.asarray(self.tray['center'])
        self.centers = dict(blue=center+(-span/2+blue_w/2)*axes[1],
                            red=center+(span/2-red_w/2)*axes[1])
        self.report.update(layout=dict(gap_m=gap, shape_allowance_m=allowance,
                                      centers={n: p.tolist() for n, p in self.centers.items()},
                                      extent_m=[length, span], order=['red', 'blue'],
                                      reason='Low cookie box first; release the taller carton above it.'),
                           layout_time=self.e.stamp())
        self.save()

    def requires(self, *measurements):
        static = [dict(evidence=m['id'], max_age_s=120) for m in self.scene_evidence]
        return static+[dict(evidence=m['id'], max_age_s=90) for m in measurements]

    def move(self, stage, plan, intent, *measurements):
        return self.call(stage, 'run', intent, plan=plan, requires=self.requires(*measurements),
                         rehearse=False, camera='side', depth=True)

    def target(self, name, fit, z):
        a, b = np.asarray(fit['size'])/2
        local = np.array([[-a, -b], [-a, b], [a, b], [a, -b]])
        xy = local @ np.asarray(self.tray['axes'])+self.centers[name]
        return np.column_stack([xy, np.full(4, z)])

    def pick_place(self, name):
        is_red = name == 'red'
        frame = self.call(name+'_frame', 'camera_frame', 'Get a fresh pickup view', camera='side', depth=True)
        measurement = self.call(name+'_pickup', 'select_target', 'Measure the current snack pose',
                                frame=frame['frame'], target=name,
                                box=[530, 295, 685, 435] if is_red else [735, 365, 925, 500])
        fit = rectangle(self.points(measurement), prism=True)
        if np.max(np.abs(np.asarray(fit['size'])-self.fits[name]['size'])) > .008:
            raise ValueError('Pickup footprint disagrees with the unobstructed wrist observation')
        self.report['objects'][name] = obj = dict(pickup_fit=fit, pickup_evidence=measurement['id'])
        height = fit['top']-self.counter
        grasp = np.r_[fit['center'], fit['top']-.011]
        jaws = np.r_[np.asarray(fit['axes'])[1], 0]
        width = fit['size'][1]*1000
        opening = min(80, width+18)
        approach = [dict(do='gripper', aperture_mm=opening)]
        if is_red:
            approach.append(dict(do='move_to', to=[.32, -.10, .43],
                                 point='down', jaws=jaws.tolist(), duration=3))
        approach.append(dict(do='move_to', to=[float(grasp[0]), float(grasp[1]), .335],
                             point='down', jaws=jaws.tolist(), duration=4))
        self.move(name+'_approach', approach,
                  'Approach the measured top surface', measurement)
        self.move(name+'_grip', [dict(do='move_to', to=grasp.tolist(), duration=3),
                    dict(do='grip', expect_mm=[width-7, width+8], hold_effort=2)],
                  'Grip below the top edge and check width and effort', measurement)
        held = self.call(name+'_held', 'observe_targets', 'Measure retention before lifting',
                         targets=[name])['measurements'][0]
        assert held['valid']
        before_tool = self.e.observation(held['frame']).tool
        local_source = transform(fit['source'], np.linalg.inv(before_tool))
        self.move(name+'_lift', dict(do='line', up=.08, duration=3), 'Lift before carrying', held)
        lifted = self.call(name+'_lift_check', 'observe_targets', 'Check that the surface stayed with the gripper',
                           targets=[name])['measurements'][0]
        assert lifted['valid']
        drift = float(np.linalg.norm(np.asarray(lifted['in_tool'])-held['in_tool']))
        rise = float(lifted['surface_center'][2]-held['surface_center'][2])
        if drift > .008 or rise < .065:
            raise ValueError(f'Inconclusive grasp: offset drift {drift}, rise {rise}')
        tool = self.e.observation(lifted['frame']).tool
        source = transform(local_source, tool)
        target_top = self.floor+height
        aligned = self.e.align(source, self.target(name, fit, target_top), tool)
        if not aligned['valid']:
            raise ValueError(aligned['reason'])
        pose = np.asarray(aligned['tool_pose'])
        # align_planar preserves height; vertical placement uses measured support
        # heights and the retained top-to-tool offset, with 1 mm release clearance.
        top_offset = float(np.mean(source[:, 2])-tool[2, 3])
        release_z = target_top-top_offset+.001
        carry_z = max(.34 if is_red else .365, release_z+.065)
        point = pose[:3, :3] @ np.array([0, 0, 1.])
        jaw_direction = pose[:3, :3] @ np.array([1., 0, 0])
        obj.update(height_m=height, grip_position=grasp.tolist(), lift_rise_m=rise,
                   retention_drift_m=drift, geometry=len(self.e.geometry)-1,
                   release_z_m=release_z, source_top_offset_m=top_offset,
                   target_top_m=target_top, aligned_tool_pose=aligned['tool_pose'],
                   alignment_time=self.e.geometry[-1]['time'])
        self.save()
        self.move(name+'_carry', dict(do='move_to', to=[*pose[:2, 3].tolist(), carry_z],
                    point=point.tolist(), jaws=jaw_direction.tolist(), duration=4),
                  'Use the alignment result for tool position and orientation', lifted)
        # A small opening releases the package without swinging fingers into its neighbour.
        self.move(name+'_place', [dict(do='move_to', to=[*pose[:2, 3].tolist(), release_z], duration=3),
                    dict(do='gripper', aperture_mm=width+8), dict(do='line', up=.075, duration=3)],
                  'Lower to the measured floor, release and lift clear', lifted)
        self.move(name+'_clear', dict(do='move_to', to=[.30, -.10, .39], duration=4),
                  'Withdraw below the cabinet and uncover the lunchbox')
        self.verify(name)

    def verify(self, name, *, suffix='placed'):
        frame = self.call(name+'_'+suffix+'_frame', 'camera_frame', 'Inspect the released package',
                          camera='side', depth=True)
        captured = self.e.observation(frame['frame'])
        fit = self.report['objects'][name]['pickup_fit']
        z = self.report['objects'][name]['target_top_m']
        top = self.target(name, fit, z)
        bottom = top.copy()
        bottom[:, 2] = self.floor
        uv = captured.view.project(np.concatenate([top, bottom]))[0]
        lo = np.maximum(np.floor(uv.min(0)-7), [0, 0]).astype(int)
        hi = np.minimum(np.ceil(uv.max(0)+7), [captured.image.width-1, captured.image.height-1]).astype(int)
        measurement = self.call(name+'_'+suffix, 'select_target', 'Measure the released object',
                                frame=frame['frame'], target=name, box=[*lo.tolist(), *hi.tolist()])
        # In the packed scene, nearby rim pixels can enter the object mask.
        # Verify the visible top only; the hand has already withdrawn.
        observed = rectangle(self.points(measurement))
        error = float(np.linalg.norm(np.asarray(observed['center'])-self.centers[name]))
        height_error = abs(observed['top']-z)
        corners = np.asarray(observed['source'])[:, :2]
        local = (corners-np.asarray(self.tray['center'])) @ np.asarray(self.tray['axes']).T
        inside = bool((np.abs(local) < np.asarray(self.tray['size'])/2-.003).all())
        check = dict(measurement=measurement['id'], observed=observed, xy_error_m=error,
                     height_error_m=height_error, inside_measured_floor=inside,
                     passed=bool(error < .010 and height_error < .006 and inside))
        self.report['checks'][name+'_'+suffix] = check
        self.save()
        if not check['passed']:
            raise ValueError(f'{name}: camera placement check failed: {check}')

    def run(self, *, measure_only=False):
        self.measure()
        if measure_only:
            return 'Measured compact layout; no motion requested'
        self.call('enable', 'enable', 'Enable the simulated Panda after planning')
        for name in ('red', 'blue'):
            self.pick_place(name)
            print('PLACED', name, flush=True, file=sys.__stdout__)
        self.call('release', 'release', 'Brake the Panda after releasing both snacks')
        time.sleep(1.5)
        self.verify('red', suffix='final')
        self.verify('blue', suffix='final')
        axes = np.asarray(self.tray['axes'])
        measured_edges = {}
        for name in ('red', 'blue'):
            check = self.report['checks'][name+'_final']
            previous = self.report['checks'][name+'_placed']['observed']
            drift = float(np.linalg.norm(np.asarray(check['observed']['center'])-previous['center']))
            if drift > .008:
                raise ValueError('The final surface position changed beyond the measurement tolerance')
            check['repeat_position_difference_m'] = drift
            measured_edges[name] = np.asarray(check['observed']['source'])[:, :2] @ axes[1]
        self.report['measured_gap_m'] = float(measured_edges['red'].min()-measured_edges['blue'].max())
        if self.report['measured_gap_m'] < .004:
            raise ValueError('The measured final footprints have insufficient separation')
        self.call('final_side', 'camera_frame', 'Record the completed compact packing', camera='side', depth=True)
        self.call('final_wrist', 'camera_frame', 'Record the final wrist view', camera='wrist', depth=True)
        status = self.call('final_status', 'status', 'Verify motor power is off')
        assert not status['enabled'] and not status['power_uncertain']
        self.report['passed'] = True
        self.save()
        return 'Both snacks compactly placed; fresh RGB-D checks passed; Panda disabled'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', type=Path, default=Path('runs/robocasa-lunchbox-compact/scene'))
    parser.add_argument('--measure-only', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (args.output.parent/(args.output.name+'.log')).open('w') as log, contextlib.redirect_stdout(log):
        episode = LunchboxEpisode(args.output, args.scene, entrypoint="lunchbox/task.py")
        with contextlib.closing(episode.tracker), episode.session:
            provenance = dict(execution_mode='Agent-authored Python with fresh RGB-D; no model calls during execution',
                              policy_inputs=['RGB-D measurements', 'robot and gripper feedback'],
                              evaluator_only=['simulator object poses', 'contact audit'], scene=str(args.scene),
                              prior_context='Task developed in this conversation with user feedback')
            (episode.out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
            episode.client.record(context=provenance)
            task = PackingTask(episode)
            try:
                result = task.run(measure_only=args.measure_only)
            except BaseException as error:
                task.report['error'] = repr(error)
                task.save()
                # Simulation-only passive brake: preserve the failed state without dropping a held object.
                episode.act('stop', intent='Stop this failed simulated task')
                episode.act('release', intent='Engage the simulated Panda brake')
                episode.finish(result='FAILED: '+repr(error))
                raise
            episode.finish(result=result)
            if not args.measure_only:
                summary = json.loads((episode.out/'final.json').read_text())
                summary['object_phases'] = {obj: dict(grip=task.stages[n+'_grip'], place=task.stages[n+'_place'])
                                           for n, obj in [('blue', 'long_snack'), ('red', 'cracker_box')]}
                (episode.out/'final.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
