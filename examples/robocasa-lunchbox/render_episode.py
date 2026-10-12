"""Render recorded physics and real world-use returns in one annotated kitchen view.

Overlays are editorial projections of saved measurements, never simulated policy inputs.
Placement outlines are proposed targets. The carry arc follows recorded tool kinematics,
assuming the retained grasp; it is not a new object observation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from world_use.bodies.mujoco_render import CameraRenderer
from world_use.cameras import View

W, H, FPS = 1920, 1080, 30
SCENE_H, SCENE_Y = H, 0
INTRO_TRIM = 3
CARD_TOP, CARD_BOTTOM = 24, 944
CARD_LEFT, CARD_RIGHT = W//4, 3*W//4
PAPER, INK, MUTED = '#f8f7f3', '#263331', '#7b827e'
MINT, GOLD = '#9fe9d5', '#c07d33'
TEAL = '#257e6c'
BLUE, BLUE_LIGHT = '#466aa5', '#b8ceff'


@lru_cache
def scene_grade():
    """Presentation white balance and midtone lift; original RGB-D evidence stays untouched."""
    ramp = np.arange(256)/255
    return np.concatenate([np.round(np.clip(ramp*gain, 0, 1)**.94*255)
                           for gain in (.985, 1.025, 1.10)]).astype('uint8').tolist()


@lru_cache
def font(size, mono=False):
    return ImageFont.truetype('/System/Library/Fonts/' + ('SFNSMono.ttf' if mono else 'SFNS.ttf'), size)


def ease(x):
    x = np.clip(x, 0, 1)
    return float(x*x*(3-2*x))


def hull(points):
    """Convex outline of the visible measured samples, not a hidden-object silhouette."""
    pts = sorted(set(map(tuple, np.asarray(points).round(2))))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    halves = []
    for seq in (pts, pts[::-1]):
        half = []
        for p in seq:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.extend(half[:-1])
    return halves


def beats(calls, geometry, task):
    out = []
    stages = task['stages']
    def add(stage, mode, chapter, request='', reply='', duration=None, name=None, start=None):
        call = stages[stage]
        c = calls[call]
        motion = duration is None
        t = c['end'] if start is None else start
        out.append(dict(call=call, mode=mode, chapter=chapter, request=request, reply=reply,
                        duration=(c['end']-c['start'])/2 if motion else duration,
                        start=c['start'] if motion else t, end=c['end'] if motion else t,
                        motion=motion, name=name))
    add('status', 'intro', 'Franka Panda / MuJoCo kitchen', duration=6.5-INTRO_TRIM)
    add('blue_initial', 'select', 'Measure the snacks', 'select_target(frame, box)',
        'EdgeTAM + depth → surface points', 2.2, 'blue')
    add('red_initial', 'select', 'Measure the snacks', 'select_target(frame, box)',
        'Position, size and orientation', 1.5, 'red')
    add('floor', 'floor', 'Measure the lunchbox', 'measure_pixels(frame, box)',
        f'Floor height → {task["floor_m"]:.3f} m', 2.2)
    add('counter', 'counter', 'Measure the support', 'measure_pixels(frame, box)',
        'Object height = top − support', 1.6)  # noqa: RUF001 — mathematical minus
    size = np.asarray(task['tray']['size'])*100
    add('tray_initial', 'layout', 'Choose a compact arrangement', 'fit surfaces → choose layout + order',
        f'Interior {size[0]:.1f} × {size[1]:.1f} cm', 4, start=task['layout_time'])  # noqa: RUF001
    for name, label in [('red', 'Cookies'), ('blue', 'Carton')]:
        obj = task['objects'][name]
        if name == 'blue':
            add(name+'_pickup', 'select', 'Measure the next pickup', 'select_target(fresh_frame, box)',
                'Fresh pose before pickup', 1.3, name)
        add(name+'_approach', 'plain', 'Pick up / '+label, 'run([open, approach])', name=name)
        add(name+'_grip', 'plain', 'Pick up / '+label, 'run([lower, grip])', name=name)
        add(name+'_lift', 'plain', 'Lift / '+label, 'run(line(up=0.08))', name=name)
        add(name+'_lift_check', 'lift', 'Check the grasp', 'observe_targets()',
            f'Lift {obj["lift_rise_m"]*100:.1f} cm · retained', 1.8, name)
        g = geometry[obj['geometry']]
        add(name+'_lift_check', 'alignment', 'Calculate the tool pose', 'align_planar(source, target, tool)',
            f'Yaw {g["result"]["yaw_delta_deg"]:+.1f}° → move_to', 2.1, name, obj['alignment_time'])
        add(name+'_carry', 'goal', 'Align / '+label, 'run(move_to(aligned_pose))', name=name)
        add(name+'_place', 'goal', 'Place / '+label, 'run([lower, release, lift])', name=name)
        add(name+'_clear', 'plain', 'Withdraw', 'run(move_to(clear_view))', name=name)
        add(name+'_placed', 'verify', 'Check the placement', 'select_target(fresh_frame, box)',
            'Position + height checked', 1.4, name)
    add('final_side', 'final', 'Complete', 'Fresh RGB-D checks → power off',
        '2 / 2 placed and verified', 4)
    return out


class Film:
    def __init__(self, root):
        self.root = root
        self.calls = json.loads((root/'calls.json').read_text())
        self.geometry = json.loads((root/'geometry.json').read_text())
        self.task = json.loads((root/'task.json').read_text())
        assert self.task['passed']
        self.stages = self.task['stages']
        self.replay = np.load(root/'replay.npz')
        self.model = mj.MjModel.from_binary_path(str(root/'scene.mjb'))
        self.model.vis.global_.offwidth = W
        self.model.vis.global_.offheight = H
        self.model.vis.quality.offsamples = 8
        wall = self.model.geom('wall_right_room_g0_vis').matid[0]
        self.model.mat_specular[wall] = 0
        self.data = mj.MjData(self.model)
        self.data.qpos[:] = self.replay['qpos'][0]
        mj.mj_forward(self.model, self.data)
        self.base = np.eye(4)
        self.base[:3, :3] = self.data.body('robot0_link0').xmat.reshape(3, 3)
        self.base[:3, 3] = self.data.body('robot0_link0').xpos
        cameras = json.loads((root/'cameras.json').read_text())
        self.hero_view = View.from_dict(cameras['side']).scaled(W, SCENE_H)
        self.view = self.hero_view
        self.renderer = CameraRenderer()
        self.point_cache, self.render_cache = {}, {}
        self.edit = beats(self.calls, self.geometry, self.task)
        self.events = [json.loads(x) for x in (root/'flight/events.jsonl').read_text().splitlines()]
        self.step_names, self.step_times = {}, {}
        for stage, names in [('red_approach', ['Open', 'Approach', 'Lower']),
                             ('blue_approach', ['Open', 'Approach']),
                             ('red_grip', ['Lower', 'Grip']), ('blue_grip', ['Lower', 'Grip']),
                             ('red_place', ['Lower', 'Release', 'Lift']),
                             ('blue_place', ['Lower', 'Release', 'Lift'])]:
            call = self.stages[stage]
            self.step_names[call] = names
            job = self.calls[call]['result']['id']
            begin = next(e for e in self.events if e['kind']=='submitted' and e['data']['job']==job)
            end = next(e for e in self.events if e['kind']=='finished' and e['data']['job']==job)
            steps = [e for e in self.events if e['kind']=='step' and begin['seq'] < e['seq'] < end['seq']]
            assert len(steps) == len(names)
            self.step_times[call] = [self.calls[call]['start']+e['t']-begin['t'] for e in steps]

    def points(self, call):
        if call not in self.point_cache:
            m = self.result(call)
            p = np.load(self.root/'flight/perception'/f'{m["id"]}.npz')['points']
            if call == self.stages['floor']:
                p = p[np.abs(p[:, 2]-self.task['floor_m']) < .001]
            self.point_cache[call] = p
        return self.point_cache[call]

    def captured(self, beat):
        result = self.result(beat['call'])
        frame = result['frame']
        frame = frame['id'] if isinstance(frame, dict) else frame
        if frame not in self.render_cache:
            raw = Image.open(self.root/'frames'/f'{frame}.png').convert('RGB')
            view = View.from_dict(json.loads((self.root/'frames'/f'{frame}.json').read_text()))
            depth = np.load(self.root/'frames'/f'{frame}.npz')['depth']
            camera = result.get('camera', self.calls[beat['call']]['arguments'].get('camera', 'side'))
            vertical = .1 if camera == 'wrist' else .5
            pic = ImageOps.fit(raw, (W, SCENE_H), method=Image.Resampling.LANCZOS, centering=(.5, vertical))
            dist = np.asarray(ImageOps.fit(Image.fromarray(depth), (W, SCENE_H),
                              method=Image.Resampling.NEAREST, centering=(.5, vertical)))
            view = view.scaled(W, SCENE_H)
            view = replace(view, cy=view.cy+(raw.height*W/raw.width-SCENE_H)*(.5-vertical))
            self.render_cache[frame] = pic, dist, view, camera
        return self.render_cache[frame]

    def outline(self, overlay, points, *, floor=False, color=BLUE_LIGHT):
        points = np.array(points).copy()
        z = self.task['floor_m']+.0006
        if floor:
            points[:, 2] = z
        uv = self.project(points)
        layer = Image.new('RGBA', (W, H))
        d = ImageDraw.Draw(layer)
        d.polygon(list(map(tuple, uv)), fill=(142, 177, 242, 38))
        d.line(list(map(tuple, np.vstack([uv, uv[0]]))), fill=color, width=3, joint='curve')
        if floor:
            pixels = np.array(layer)
            ys, xs = np.nonzero(pixels[:, :, 3])
            keep = (ys >= SCENE_Y) & (ys < SCENE_Y+SCENE_H)
            ys, xs = ys[keep], xs[keep]
            rays = self.view.unproject(np.column_stack([xs+.5, ys-SCENE_Y+.5]), np.ones(len(xs)))
            rays -= self.view.T[:3, 3]
            distance = (z-self.view.T[2, 3])/rays[:, 2]
            hidden = self.depth[ys-SCENE_Y, xs] < distance-.002
            pixels[ys[hidden], xs[hidden], 3] = 0
            layer = Image.fromarray(pixels)
        overlay.alpha_composite(layer)
        return uv.mean(0)

    def target(self, name, *, initial=False):
        if initial:
            a, b = np.asarray(self.task['fits'][name]['size'])/2
            xy = np.array([[-a, -b], [-a, b], [a, b], [a, -b]]) @ np.asarray(self.task['tray']['axes'])
            xy += self.task['layout']['centers'][name]
            return np.column_stack([xy, np.full(4, self.task['floor_m'])])
        obj = self.task['objects'][name]
        return np.asarray(self.geometry[obj['geometry']]['target'])

    def draw(self, beat, local):
        mode, name = beat['mode'], beat['name']
        t = beat['start']+(beat['end']-beat['start'])*local/max(beat['duration'], 1/FPS)
        ix = int(np.clip(np.searchsorted(self.replay['time'], t), 0, len(self.replay['time'])-1))
        self.data.qpos[:] = self.replay['qpos'][ix]
        mj.mj_forward(self.model, self.data)
        if mode == 'intro':
            u = ease((local+INTRO_TRIM-2)/4)
            eye = (1-u)*np.array([-.20, 1.45, 1.20])+u*np.array([-.15, 1.25, 1.14])
            target = (1-u)*np.array([.35, .02, .40])+u*np.array([.39, .01, .38])
            self.view = View.look_at(eye, target, fov_deg=61-5*u, size=(W, SCENE_H))
            camera = ''
        else:
            self.view = self.hero_view
            camera = 'Side camera · motion replay'
        if beat['motion'] or mode == 'intro':
            picture, self.depth = self.renderer.render(self.model, self.data,
                replace(self.view, T=self.base@self.view.T), depth=True)
        else:
            picture, self.depth, self.view, source = self.captured(beat)
            camera = source.capitalize()+' camera · captured RGB-D'
        canvas = Image.new('RGB', (W, H), PAPER)
        canvas.paste(picture.point(scene_grade()), (0, SCENE_Y))
        overlay = Image.new('RGBA', (W, H))
        if mode == 'intro':
            for stage, y, title, subtitle in [('side', 126, 'Side camera', 'Fixed beside the counter'),
                                             ('wrist', 454, 'Wrist camera', 'Moves with the hand')]:
                frame = self.result(self.stages[stage])['frame']
                thumb = Image.open(self.root/'frames'/f'{frame}.png').convert('RGB').resize(
                    (320, 240), Image.Resampling.LANCZOS).point(scene_grade())
                d = ImageDraw.Draw(overlay)
                d.rounded_rectangle((32, y, 360, y+304), radius=20, fill=(248, 247, 243, 246))
                mask = Image.new('L', thumb.size)
                ImageDraw.Draw(mask).rounded_rectangle((0, 0, 319, 239), radius=16, fill=255)
                overlay.paste(thumb, (36, y+4), mask)
                d.text((48, y+253), title, font=font(22), fill=TEAL)
                d.text((48, y+281), subtitle, font=font(15), fill=MUTED)
        elif mode == 'select':
            p = self.cloud(overlay, beat['call'], local/1.1)
            self.label(overlay, p, 'Carton' if name=='blue' else 'Cookies', 'EdgeTAM + depth',
                       offset=(85, -105) if name=='blue' else (-300, -25))
        elif mode == 'floor':
            p = self.cloud(overlay, beat['call'], local)
            self.label(overlay, p, 'Measured floor', 'sets the release height', offset=(200, -55))
        elif mode == 'counter':
            p = self.cloud(overlay, beat['call'], local)
            self.label(overlay, p, 'Measured counter', 'reference for object height', offset=(-400, -20))
        elif mode == 'layout':
            for n in ('red', 'blue'):
                self.outline(overlay, self.task['fits'][n]['source'])
                p = self.outline(overlay, self.target(n, initial=True), floor=True)
                number = '1' if n=='red' else '2'
                x, y = p
                d = ImageDraw.Draw(overlay)
                d.ellipse((x-18, y-18, x+18, y+18), fill=PAPER, outline=BLUE, width=2)
                d.text((x-8, y-14), number, font=font(25), fill=BLUE)
            p = self.project([[*self.task['tray']['center'], self.task['floor_m']]])[0]
            self.label(overlay, p, 'Cookies first. Carton second.', 'room for release and withdrawal',
                       offset=(230, -20), color=BLUE)
        elif mode == 'lift':
            before = self.result(self.stages[name+'_held'])
            after = self.result(beat['call'])
            a, b = np.array(before['surface_center']), np.array(after['surface_center'])
            self.cloud(overlay, beat['call'])
            p, q = self.project([a, b])
            d = ImageDraw.Draw(overlay)
            d.line([tuple(p), tuple(q)], fill=MINT, width=3)
            for x, y in (p, q):
                d.line((x-8, y, x+8, y), fill=TEAL, width=3)
            self.label(overlay, q, f'+{100*(b[2]-a[2]):.1f} cm', 'surface stayed with gripper', offset=(-370, 5))
        elif mode == 'alignment':
            g = self.geometry[self.task['objects'][name]['geometry']]
            pts = np.asarray(g['source'])
            self.outline(overlay, pts)
            self.outline(overlay, g['target'], floor=True)
            self.rotation(overlay, g, pts.mean(0), ease(local/.9))
        elif mode == 'goal':
            self.outline(overlay, self.target(name), floor=True)
            if beat['call'] == self.stages[name+'_carry']:
                g = self.geometry[self.task['objects'][name]['geometry']]
                site = self.data.site('gripper0_right_grip_site')
                tool = np.eye(4)
                tool[:3, :3], tool[:3, 3] = site.xmat.reshape(3, 3), site.xpos
                tool = np.linalg.inv(self.base) @ tool
                held = tool @ np.linalg.inv(g['tool'])
                center = held[:3, :3] @ np.mean(g['source'], axis=0) + held[:3, 3]
                self.rotation(overlay, g, center)
            elif beat['call'] == self.stages[name+'_place']:
                obj = self.task['objects'][name]
                p = self.project([[*self.task['layout']['centers'][name], obj['release_z_m']]])[0]
                self.label(overlay, p, f'Release at {obj["release_z_m"]:.3f} m',
                           'from floor + held geometry', offset=(-410, 20))
        elif mode == 'verify':
            observed = self.task['checks'][name+'_placed']['observed']
            p = self.outline(overlay, observed['source'], color=MINT)
            self.label(overlay, p, 'Placement checked', 'fresh RGB-D measurement', offset=(-390, 35))
        elif mode == 'final':
            for n in ('red', 'blue'):
                observed = self.task['checks'][n+'_final']['observed']
                x, y = self.project([np.mean(observed['source'], axis=0)])[0]
                d = ImageDraw.Draw(overlay)
                d.ellipse((x-15, y-15, x+15, y+15), fill=PAPER, outline=TEAL, width=2)
                d.line((x-7, y, x-1, y+6, x+9, y-7), fill=TEAL, width=3)
            self.label(overlay, self.project([[*self.task['tray']['center'], self.task['floor_m']]])[0],
                       '2 / 2 packed', 'released · checked · power off', offset=(-395, 50))
        clip = ImageDraw.Draw(overlay)
        clip.rectangle((0, 0, W, 111), fill=(0, 0, 0, 0))
        clip.rectangle((0, CARD_BOTTOM-12, W, H), fill=(0, 0, 0, 0))
        carry = mode=='goal' and beat['call']==self.stages[name+'_carry']
        opacity = 1 if mode=='intro' or carry else ease(local/.3)
        overlay.putalpha(overlay.getchannel('A').point(lambda v: int(v*opacity)))
        canvas = Image.alpha_composite(canvas.convert('RGBA'), overlay).convert('RGB')
        d = ImageDraw.Draw(canvas)
        for top, bottom in ((CARD_TOP, 96), (CARD_BOTTOM, H-24)):
            d.rounded_rectangle((CARD_LEFT, top, CARD_RIGHT, bottom), radius=26, fill=PAPER)
        d.rounded_rectangle((CARD_LEFT+24, 44, CARD_LEFT+30, 77), radius=3, fill=TEAL)
        d.text((CARD_LEFT+46, 39), 'world-use', font=font(33), fill=INK)
        d.text((CARD_LEFT+245, 49), camera, font=font(18), fill=TEAL)
        heading = beat['chapter']
        d.text((CARD_RIGHT-24-d.textlength(heading, font=font(24)), 46), heading, font=font(24), fill=MUTED)
        footer = CARD_BOTTOM
        left, divider, right = CARD_LEFT+24, CARD_LEFT+536, CARD_LEFT+560
        if mode == 'intro':
            d.text((left, footer+32), 'Fit both snacks in the lunchbox.', font=font(36), fill=INK)
            subtitle = 'Measure. Arrange. Verify.'
            d.text((CARD_RIGHT-24-d.textlength(subtitle, font=font(21)), footer+42),
                   subtitle, font=font(21), fill=MUTED)
        else:
            owner = 'Agent-written Python' if mode=='layout' else 'Task Python → world-use'
            owner_color = BLUE if mode=='layout' else TEAL
            d.rounded_rectangle((left, footer+23, left+5, footer+42), radius=2, fill=owner_color)
            d.text((left+18, footer+22), owner, font=font(17), fill=owner_color)
            command = beat['request']
            d.text((left, footer+59), command, font=font(21, True), fill=INK)
            d.line((divider, footer+23, divider, footer+89), fill='#d9ddd5', width=1)
            sequence = beat['motion'] and beat['call'] in self.step_names
            caption = (f'{len(self.step_names[beat["call"]])} steps · execution at 100 Hz' if sequence
                       else 'Executing' if beat['motion'] else 'Calculated from measurements' if mode=='layout'
                       else 'Returned measurement' if mode in ('floor', 'counter', 'select', 'lift', 'verify')
                       else 'Calculated pose' if mode=='alignment' else 'Task result')
            d.text((right, footer+22), caption, font=font(17), fill=MUTED)
            if sequence:
                self.execution(d, beat, t, right)
            else:
                d.text((right, footer+60), beat['reply'], font=font(23), fill=INK)
            assert d.textlength(command, font=font(21, True)) < divider-left-24
            assert sequence or d.textlength(beat['reply'], font=font(23)) < CARD_RIGHT-right-24
        return canvas

    def rotation(self, overlay, geometry, center, progress=1):
        yaw = np.radians(geometry['result']['yaw_delta_deg'])
        # Put the positive-yaw arc on the visible side of the low cookie box.
        start = (np.pi if yaw > 0 else 0)-yaw
        angles = np.linspace(start, start+yaw*progress, 60)
        arc = center+np.column_stack([.085*np.cos(angles), .085*np.sin(angles), np.full(60, -.006)])
        uv = self.project(arc)
        d = ImageDraw.Draw(overlay)
        d.line(list(map(tuple, uv)), fill=PAPER, width=7, joint='curve')
        d.line(list(map(tuple, uv)), fill=TEAL, width=3, joint='curve')
        direction = uv[-1]-uv[-3]
        if np.linalg.norm(direction) > .01:
            direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            tip = uv[-1]
            d.polygon([tuple(tip+2*direction), tuple(tip-15*direction+8*normal),
                       tuple(tip-15*direction-8*normal)], fill=PAPER)
            d.polygon([tuple(tip), tuple(tip-13*direction+5*normal),
                       tuple(tip-13*direction-5*normal)], fill=TEAL)
        self.label(overlay, self.project([center])[0], f'{abs(np.degrees(yaw)):.0f}° rotation',
                   'alignment result drives the motion', offset=(-410, -30))

    def result(self, call):
        r = self.calls[call]['result']
        return r['measurements'][0] if 'measurements' in r else r

    def project(self, points):
        uv = self.view.project(points)[0]
        uv[:, 1] += SCENE_Y
        return uv

    def label(self, overlay, anchor, title, subtitle='', *, offset=(50, -95), color=TEAL):
        d = ImageDraw.Draw(overlay)
        x, y = np.asarray(anchor) + np.asarray(offset)*1.2
        tw = max(d.textlength(title, font=font(29)), d.textlength(subtitle, font=font(18)))
        width, height = tw+36, 83 if subtitle else 56
        x = float(np.clip(x, 24, W-width-24))
        y = float(np.clip(y, 120, CARD_BOTTOM-height-24))
        ax, ay = map(float, anchor)
        ex = x if ax < x else x+width if ax > x+width else ax
        ey = y+height/2
        d.line([(ax, ay), (ex, ey)], fill=(249, 255, 250, 225), width=3)
        d.line([(ax, ay), (ex, ey)], fill=color, width=1)
        d.ellipse((ax-4, ay-4, ax+4, ay+4), fill=color, outline=PAPER, width=1)
        d.rounded_rectangle((x+1, y+3, x+width+1, y+height+3), radius=9, fill=(22, 35, 30, 22))
        d.rounded_rectangle((x, y, x+width, y+height), radius=8, fill=(251, 251, 247, 241),
                            outline=(255, 255, 255, 220), width=1)
        d.text((x+18, y+12), title, font=font(29), fill=color)
        if subtitle:
            d.text((x+18, y+50), subtitle, font=font(18), fill=MUTED)

    def execution(self, draw, beat, stamp, x):
        call = beat['call']
        names = self.step_names[call]
        active = int(np.clip(np.searchsorted(self.step_times[call], stamp, side='right')-1, 0, len(names)-1))
        y = CARD_BOTTOM+60
        for i, name in enumerate(names):
            width = draw.textlength(name, font=font(25))
            draw.text((x, y), name, font=font(25), fill=TEAL if i==active else MUTED)
            if i == active:
                draw.line((x, y+36, x+width, y+36), fill=TEAL, width=2)
            x += width+26
            if i+1 < len(names):
                draw.line((x, y+11, x+5, y+16, x, y+21), fill='#b7bcb6', width=1)
                x += 35
        if x-26 > CARD_RIGHT-24:
            raise ValueError('Step sequence exceeds the footer')

    def cloud(self, overlay, call, progress=1):
        uv = self.project(self.points(call))
        poly = hull(uv)
        d = ImageDraw.Draw(overlay)
        d.polygon(poly, fill=(159, 233, 213, 18))
        d.line(poly+[poly[0]], fill=MINT, width=2, joint='curve')
        ordered = uv[np.argsort(uv[:, 1])]
        for x, y in ordered[:int(len(ordered)*ease(progress))][::17]:
            d.ellipse((x-.9, y-.9, x+.9, y+.9), fill=(211, 255, 240, 180))
        return self.project([self.result(call)['surface_center']])[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--stills', action='store_true')
    parser.add_argument('--output-name', default='kitchen-tool-use-v8')
    args = parser.parse_args()
    film = Film(args.recording)
    (args.recording/f'{args.output_name}-edit.json').write_text(json.dumps(film.edit, indent=2)+'\n')
    encoder = None
    if not args.stills:
        encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{W}x{H}', '-framerate', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'slow', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(args.recording/f'{args.output_name}.mp4')], stdin=subprocess.PIPE)
    try:
        for i, beat in enumerate(film.edit):
            count = max(1, round(beat['duration']*FPS))
            for f in ([count//2] if args.stills else range(count)):
                image = film.draw(beat, f/FPS)
                if f == count//2:
                    image.save(args.recording/f'{args.output_name}-{i:02d}.jpg', quality=95)
                if encoder:
                    encoder.stdin.write(image.tobytes())
            print(f'{i+1}/{len(film.edit)} {beat["mode"]}', flush=True)
    finally:
        film.renderer.close()
        if encoder:
            encoder.stdin.close()
            code = encoder.wait()
            if code:
                raise RuntimeError(f'ffmpeg exited {code}')


if __name__ == '__main__':
    main()
