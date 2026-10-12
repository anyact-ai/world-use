"""Record model-selected world-use calls with a fixed panel view and Panda wrist camera."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'robocasa-counter'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'robocasa-lunchbox'))
from agent_episode import Episode
from sensor_views import WristCamera

from world_use.cameras import SimCamera, View


class DialEpisode(Episode):
    def __init__(self, output, scene=Path('runs/robocasa-dial/scene')):
        rig = json.loads((scene/'metadata.json').read_text())['rig']
        def cameras(body):
            v = View.look_at(**rig['side'], size=(1600, 1000))
            # Rig setup is world-frame calibration, public frames use robot base.
            from dataclasses import replace
            v = replace(v, T=np.linalg.inv(body.base) @ v.T)
            return dict(side=SimCamera('side', v, body), wrist=WristCamera(body))
        super().__init__(output, scene, park_left_m=0, camera_factory=cameras,
                         task='Set the oven temperature to 180 C; leave the timer and burners unchanged.',
                         setup='Original six-control RoboCasa stove; passively hinged contact-driven knobs',
                         sensors='Fixed panel RGB-D and Panda eye-in-hand RGB-D; EdgeTAM CPU')
        self.notes = []

    def observation(self, frame_id):
        frame = self.client.frame(id=frame_id, depth=True)
        path = self.out/'frames'
        path.mkdir(exist_ok=True)
        frame.image.save(path/(frame.id+'.png'))
        np.savez_compressed(path/(frame.id+'.npz'), depth=frame.depth, tool=frame.tool)
        (path/(frame.id+'.json')).write_text(json.dumps(frame.view.to_dict(), indent=2)+'\n')
        return frame

    def note(self, **data):
        self.notes.append(dict(time=self.stamp(), **data))
        (self.out/'decisions.json').write_text(json.dumps(self.notes, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scene', type=Path, default=Path('runs/robocasa-dial/scene'))
    args = parser.parse_args()
    e = DialEpisode(args.output, args.scene)
    act, observation, note, finish = e.act, e.observation, e.note, e.finish
