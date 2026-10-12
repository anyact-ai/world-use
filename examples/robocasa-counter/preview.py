"""Inspect an upstream RoboCasa counter-to-sink task with its Franka Panda.

This is an environment preview, not an agent rollout or a world-use adapter.
Run with the isolated RoboCasa interpreter documented beside this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import robocasa
import robosuite
from PIL import Image
from robocasa.environments.kitchen.atomic.kitchen_pick_place import PickPlaceCounterToSink
from robosuite.controllers import load_composite_controller_config


class PandaCupToSink(PickPlaceCounterToSink):
    """A small cup that fits the Panda's 80 mm jaw travel."""

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs[0]['object_scale'] = .75
        return cfgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/robocasa-small-cup/scene'))
    parser.add_argument('--layout', type=int, default=1)
    parser.add_argument('--style', type=int, default=2)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = robosuite.make(
        'PandaCupToSink', robots='PandaOmron', obj_groups='mug', obj_registries=('objaverse',),
        controller_configs=load_composite_controller_config(robot='PandaOmron'),
        layout_ids=[args.layout], style_ids=[args.style], seed=args.seed,
        generative_textures=None, translucent_robot=False, randomize_cameras=False,
        update_fxtr_cfg_dict={name: {'enable': False} for name in (
            'knife_block', 'digital_scale', 'paper_towel', 'oil_bottle', 'soap_dispenser',
            'glass_cup', 'glass_cup_2', 'glass_cup_3',
        )},
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, use_object_obs=False,
        control_freq=20, ignore_done=True,
    )
    renderer = None
    try:
        env.reset()
        # Zero Cartesian/base action holds the existing pose while rigid objects settle.
        action = np.zeros(env.action_dim)
        for _ in range(30):
            env.step(action)
        model, data = env.sim.model._model, env.sim.data._data
        model.vis.global_.offwidth = 1600
        model.vis.global_.offheight = 1000
        model.vis.quality.shadowsize = 4096
        model.vis.quality.offsamples = 4
        renderer = mujoco.Renderer(model, height=1000, width=1600)
        option = mujoco.MjvOption()
        option.geomgroup[0] = False  # robosuite's collision meshes
        option.sitegroup[:] = False
        sink = env.sink.pos
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = sink + np.array([0, -.35, .10])
        camera.distance = 2.5
        camera.azimuth = 125
        camera.elevation = -22
        renderer.update_scene(data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render()).save(args.output / 'overview.png')
        renderer.update_scene(data, camera='robot0_agentview_center', scene_option=option)
        Image.fromarray(renderer.render()).save(args.output / 'camera.png')
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        np.savez_compressed(args.output / 'depth.npz', depth_m=depth)
        metadata = {
            'status': 'Environment preview only; no LLM or world-use control has run in this scene',
            'task': env.get_ep_meta()['lang'], 'environment': 'PickPlaceCounterToSink',
            'robot': 'Franka Panda on the upstream Omron base; no base commands issued',
            'layout': args.layout, 'style': args.style, 'seed': args.seed,
            'target_object_scale': .75,
            'versions': {'robocasa': robocasa.__version__, 'robosuite': robosuite.__version__,
                         'mujoco': mujoco.__version__},
            'physics_seconds': float(data.time),
            'finite_state': bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            'source': 'https://github.com/robocasa/robocasa/tree/456174f62b89b8fca99eaaf33949c29fec9cfc2a',
        }
        mujoco.mj_saveModel(model, str(args.output / 'scene.mjb'), None)
        mujoco.mj_saveLastXML(str(args.output / 'scene.xml'), model)
        np.savez_compressed(args.output / 'state.npz', qpos=data.qpos, qvel=data.qvel)
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(metadata, indent=2), flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == '__main__':
    main()
