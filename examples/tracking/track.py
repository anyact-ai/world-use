"""Track a selected object in a recording or a configured camera. Does not move a robot."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from world_use.cameras import Frame
from world_use.client import Client
from world_use.vision import EdgeTAM


def preview(frame, observation, path):
    pixels = np.array(frame.image.convert("RGB"))
    if observation.status == "tracked" and observation.mask is not None:
        mask = observation.mask
        pixels[mask] = (pixels[mask] * .55 + np.array([70, 220, 100]) * .45).astype(np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 320, 25), fill="black")
    draw.text((8, 6), f"EdgeTAM / {observation.status} / age {observation.age_s:.2f}s", fill="white")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="recorded video; needs opencv-python-headless")
    source.add_argument("--seed", type=Path, help="saved Frame.to_dict() JSON from a live daemon")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"))
    selection.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--frames", type=int, default=300, help="maximum observations, including the seed")
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--model-path", type=Path, help="local copy of the pinned checkpoint")
    parser.add_argument("--url", help="daemon URL for a live camera")
    parser.add_argument("--preview", type=Path, help="overwrite this PNG with the latest annotated frame")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    cap = None
    try:
        with EdgeTAM(device=args.device, model_path=args.model_path) as tracker:
            if args.video:
                import cv2  # only the video example needs OpenCV, for decoding

                cap = cv2.VideoCapture(str(args.video))
                if not cap.isOpened():
                    raise ValueError(f"cannot open {args.video}")

                def next_frame():
                    ok, bgr = cap.read()
                    return Frame(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), "video") if ok else None

                seed = next_frame()
                if seed is None:
                    raise ValueError("video contains no frames")
            else:
                seed = Frame.from_dict(json.loads(args.seed.read_text()))
                client = Client(args.url) if args.url else Client()

                def next_frame():
                    time.sleep(.05)                # a file camera may not have produced another frame yet
                    return client.frame(seed.camera)

            print(json.dumps(dict(device=tracker.device, camera=seed.camera)), flush=True)
            frame = seed
            last_status = None
            for index in range(args.frames):
                if index:
                    frame = next_frame()
                    if frame is None:
                        break
                observation = (tracker.select(frame, point=args.point, box=args.box) if index == 0
                               else tracker.update(frame))
                if observation.status != last_status or index % 15 == 0 or index + 1 == args.frames:
                    print(json.dumps(observation.to_dict()), flush=True)
                last_status = observation.status
                if args.preview:
                    preview(frame, observation, args.preview)
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
