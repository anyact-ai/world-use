"""Interactive world-use recording for the lunchbox demo; no prewritten pickup policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa-counter'))
from agent_episode import Episode
from sensor_views import cameras

from world_use.geometry import align_planar


class LunchboxEpisode(Episode):
    def __init__(self, output, scene=Path('runs/robocasa-lunchbox-compact/scene')):
        super().__init__(output, scene, park_left_m=0, camera_factory=cameras,
                         task='Pack both snacks in the open lunchbox; leave them released, then power off',
                         sensors='Fixed side RGB-D and Panda eye-in-hand RGB-D; EdgeTAM CPU',
                         setup='Upstream RoboCasa kitchen, two rigid packaged snacks and an open container')
        self.geometry = []
        (self.out / 'camera-rig.json').write_text(json.dumps(dict(
            side=dict(mount='fixed beside the counter', view=self.daemon.cameras['side'].view.to_dict()),
            wrist=dict(mount='robot0_right_hand', mujoco_camera='robot0_eye_in_hand',
                       per_capture_pose='frames/<id>.json')), indent=2)+'\n')

    def act(self, tool, *, intent, **arguments):
        result = super().act(tool, intent=intent, **arguments)
        if isinstance(result, dict):
            frame = result.get('frame')
            if isinstance(frame, dict):
                frame = frame.get('id')
            if frame:
                self.observation(frame)
            for measurement in result.get('measurements', []):
                if measurement.get('frame'):
                    self.observation(measurement['frame'])
        return result

    def align(self, source, target, tool, *, max_error_m=.004):
        result = align_planar(source, target, tool, max_error_m=max_error_m)
        entry = dict(time=self.stamp(), function='world_use.geometry.align_planar',
                     source=np.asarray(source).tolist(), target=np.asarray(target).tolist(),
                     tool=np.asarray(tool).tolist(), max_error_m=max_error_m, result=result)
        self.geometry.append(entry)
        (self.out / 'geometry.json').write_text(json.dumps(self.geometry, indent=2)+'\n')
        print(json.dumps(entry, indent=2), flush=True)
        return result

    def observation(self, frame_id):
        """Preserve the exact public frame, including calibration, depth and captured tool pose."""
        frame = self.client.frame(id=frame_id, depth=True)
        path = self.out / 'frames'
        path.mkdir(exist_ok=True)
        frame.image.save(path / (frame.id + '.png'))
        np.savez_compressed(path / (frame.id + '.npz'), depth=frame.depth, tool=frame.tool)
        (path / (frame.id + '.json')).write_text(json.dumps(frame.view.to_dict(), indent=2))
        return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/lunchbox-new')
    parser.add_argument('--scene', type=Path, default=Path('runs/robocasa-lunchbox-compact/scene'))
    args = parser.parse_args()
    episode = LunchboxEpisode(args.output, args.scene)
    act, align, observation, finish = episode.act, episode.align, episode.observation, episode.finish
