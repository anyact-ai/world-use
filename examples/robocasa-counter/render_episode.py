"""Edit the selected counter-to-sink recording using physics and actual tool returns."""
from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT, FPS = 1600, 900, 24
BG, INK, MUTED, ACCENT = '#f5f3ee', '#26342f', '#788079', '#218877'
SCENE_SIZE = (1120, 690)


@lru_cache
def font(size, mono=False):
    name = 'SFNSMono.ttf' if mono else 'SFNS.ttf'
    return ImageFont.truetype('/System/Library/Fonts/' + name, size)


def edit(calls):
    beats = []

    def add(call, evidence, mode, request, seconds=None):
        c = calls[call]
        moving = seconds is None
        beats.append(dict(call=call, evidence=evidence, mode=mode, request=request,
                          duration=c['end'] - c['start'] if moving else seconds,
                          start=c['start'] if moving else c['end'], end=c['end'], motion=moving))

    add(1, 1, 'points', 'select_target(target="cup", box=[240, 451, 296, 517])', 3)
    add(3, 1, 'points', 'run([gripper(...), move_to(...), move_to([.41, .39, .35])])')
    add(4, 1, 'points', 'run(move_to([.408, .388, .263]))')
    add(5, 1, 'points', 'run(gripper(aperture_mm=30))')
    add(5, 5, 'contact', 'run(gripper(aperture_mm=30))', 1.2)
    add(6, 5, 'contact', 'run(grip(expect_mm=[50, 70], hold_effort=2))')
    add(6, 6, 'grip', 'run(grip(expect_mm=[50, 70], hold_effort=2))', 1.5)
    add(8, 6, 'grip', 'run(line(up=.08, duration=3))')
    add(9, 9, 'invalid', 'observe_targets(targets=["held-cup"])', 1.4)
    add(10, 10, 'lift', 'select_target(target="held-cup", point=[405, 483])', 3)
    add(12, 12, 'floor', 'measure_pixels(target="sink-floor", box=[718, 500, 746, 521])', 1.8)
    add(13, 12, 'floor', 'run([move_to([.45, .10, .36]), move_to([.55, -.20, .36])])')
    add(15, 12, 'floor', 'run(move_to([.55, -.20, .16]))')
    add(16, 12, 'floor', 'run([move_to(...), gripper(aperture_mm=80), line(up=.16)])')
    add(17, 16, 'released', 'run(move_to([.33, .02, .42]))')
    add(21, 21, 'final', 'release(); measure_pixels(target="cup-surface")', 3)
    return beats


def measurement(call):
    result = call['result']
    return result['measurements'][0] if 'measurements' in result else result


def panel(root, calls, beat):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(canvas)
    d.text((28, 23), 'world-use', font=font(38), fill=INK)
    d.text((272, 35), 'Put the cup in the sink.', font=font(27), fill=INK)
    d.text((1380, 38), 'Franka Panda', font=font(21), fill=MUTED)
    d.line((28, 83, 1572, 83), fill='#dedfd8', width=1)
    d.rounded_rectangle((1172, 108, 1572, 794), radius=14, fill='#fffefa')
    c = calls[beat['evidence']]
    mode = beat['mode']
    titles = {
        'points': 'Cup located', 'contact': 'Jaws stopped early', 'grip': 'Grip detected',
        'invalid': 'Cup + background mixed', 'lift': 'Cup lifted',
        'floor': 'Sink floor located', 'released': 'Cup released', 'final': 'Cup is in the sink',
    }
    sources = {
        'points': 'EdgeTAM selection + camera depth', 'contact': 'Gripper feedback',
        'grip': 'Gripper feedback', 'invalid': 'Depth check', 'lift': 'Fresh selection + camera depth',
        'floor': 'Camera depth', 'released': 'Returned camera image', 'final': 'Returned camera image + depth',
    }
    actions = {
        1: 'Approach the cup', 3: 'Approach the cup', 4: 'Lower jaws around cup',
        5: 'Switch to contact grip' if mode == 'contact' else 'Close the jaws',
        6: 'Lift 8 cm' if mode == 'grip' else 'Switch to contact grip',
        8: 'Lift 8 cm', 9: 'Select the cup again', 10: 'Measure the sink floor',
        12: 'Move over the sink', 13: 'Move over the sink', 15: 'Lower into the sink',
        16: 'Open jaws, then lift away', 17: 'Move the arm clear', 21: 'Finish with power off',
    }
    color = '#966335' if mode in ('contact', 'invalid') else ACCENT
    d.text((1196, 124), 'TOOL RESULT', font=font(15), fill=MUTED)
    d.text((1196, 157), titles[mode], font=font(26), fill=INK)
    d.text((1196, 195), sources[mode], font=font(17), fill=MUTED)
    crops = {
        'points': (156, 395, 380, 567), 'contact': (303, 418, 549, 607),
        'grip': (303, 418, 549, 607), 'invalid': (302, 348, 548, 537),
        'floor': (558, 391, 834, 603), 'released': (399, 297, 665, 501),
        'final': (399, 297, 665, 501),
    }
    if mode == 'lift':
        # Identical crop and scale from the same fixed camera, before and after lifting.
        crop = (300, 365, 545, 681)
        for call, x, label in ((calls[7], 1196, 'Before'), (c, 1380, 'After')):
            im = Image.open(call['images'][0]).convert('RGB').crop(crop)
            canvas.paste(im.resize((168, 217), Image.Resampling.LANCZOS), (x, 260))
            d.text((x, 232), label, font=font(18), fill=MUTED)
    else:
        im = Image.open(c['images'][0]).convert('RGB').crop(crops[mode])
        canvas.paste(im.resize((352, 270), Image.Resampling.LANCZOS), (1196, 232))
    if mode == 'points':
        xyz = measurement(c)['surface_center']
        d.text((1196, 526), 'Position from the robot base', font=font(18), fill=MUTED)
        for x, label, value in zip((1196, 1320, 1438), ('Forward', 'Left', 'Up'), xyz, strict=True):
            d.text((x, 562), label, font=font(17), fill=MUTED)
            d.text((x, 590), f'{100 * value:.1f}', font=font(28), fill=ACCENT)
            d.text((x + 64, 600), 'cm', font=font(15), fill=MUTED)
    elif mode == 'grip':
        aperture = c['result']['outcome']['data']['aperture_mm']
        d.text((1196, 526), f'{aperture:.1f} mm', font=font(35), fill=ACCENT)
        d.text((1196, 575), 'Jaw opening at contact', font=font(20), fill=INK)
        d.text((1196, 610), 'Expected: 50-70 mm', font=font(18), fill=MUTED)
    elif mode == 'lift':
        rise = 100 * (measurement(c)['surface_center'][2] - calls[7]['result']['surface_center'][2])
        d.text((1196, 526), f'+{rise:.1f} cm', font=font(35), fill=ACCENT)
        d.text((1196, 577), 'Cup surface rose with the jaws', font=font(20), fill=INK)
    elif mode == 'contact':
        d.text((1196, 526), 'Asked: 30 mm', font=font(24), fill=INK)
        d.text((1196, 566), 'Reached: 63 mm', font=font(24), fill=color)
        d.text((1196, 610), 'world-use stopped the close', font=font(18), fill=MUTED)
    elif mode == 'invalid':
        d.text((1196, 526), 'No reliable cup position', font=font(23), fill=color)
        d.text((1196, 571), 'world-use rejected the depth', font=font(18), fill=MUTED)
    elif mode == 'floor':
        z = measurement(c)['surface_center'][2]
        d.text((1196, 526), f'{100 * z:.1f} cm', font=font(35), fill=ACCENT)
        d.text((1196, 575), 'Sink floor above the robot base', font=font(19), fill=INK)
        d.text((1196, 610), 'Height used to plan the descent', font=font(18), fill=MUTED)
    else:
        d.text((1196, 526), 'Resting on its side', font=font(26), fill=ACCENT)
        d.text((1196, 574), 'Gripper open and clear', font=font(20), fill=INK)
    # These labels summarize recorded calls, not invented model thoughts.
    d.line((1196, 659, 1548, 659), fill='#e0e4dc', width=1)
    d.line((1210, 681, 1210, 708), fill=color, width=2)
    d.line((1204, 702, 1210, 708, 1216, 702), fill=color, width=2)
    d.text((1230, 684), 'LLM ACTION', font=font(15), fill=MUTED)
    d.text((1196, 736), actions[beat['call']], font=font(24), fill=INK)
    d.text((28, 809), 'LLM → world-use', font=font(17), fill=MUTED)
    if d.textlength(beat['request'], font=font(21, True)) > 1120:
        raise ValueError('Command exceeds available width: ' + beat['request'])
    d.text((28, 839), beat['request'], font=font(21, True), fill=INK)
    d.text((28, 880), 'RoboCasa / MuJoCo · development run · pauses cut · condensed calls',
           font=font(14), fill=MUTED)
    d.text((1432, 880), 'motion 1x' if beat['motion'] else 'frame held', font=font(14), fill=MUTED)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--stills', action='store_true')
    parser.add_argument('--output-name', default='panda-world-use')
    args = parser.parse_args()
    root = args.recording
    calls = json.loads((root / 'calls.json').read_text())
    # This edit uses the selected episode's call order; fail rather than silently mislabel another run.
    assert calls[21]['arguments'].get('target') == 'cup-surface'
    replay = np.load(root / 'replay.npz')
    model = mujoco.MjModel.from_binary_path(str(root / 'scene.mjb'))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=SCENE_SIZE[1], width=SCENE_SIZE[0])
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    option.sitegroup[:] = False
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.12, -.48, 1.02]
    camera.distance, camera.azimuth, camera.elevation = 1.75, 60, -32
    beats = edit(calls)
    (root / (args.output_name + '-edit.json')).write_text(json.dumps(beats, indent=2) + '\n')
    encoder = None
    if not args.stills:
        encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{WIDTH}x{HEIGHT}', '-framerate', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(root / (args.output_name + '.mp4'))], stdin=subprocess.PIPE)
    try:
        for i, beat in enumerate(beats):
            template = panel(root, calls, beat)
            count = round(beat['duration'] * FPS)
            for f in ([count // 2] if args.stills else range(count)):
                t = beat['start'] + (beat['end'] - beat['start']) * f / max(1, count - 1)
                index = int(np.clip(np.searchsorted(replay['time'], t), 0, len(replay['time']) - 1))
                data.qpos[:] = replay['qpos'][index]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                canvas = template.copy()
                canvas.paste(Image.fromarray(renderer.render()), (28, 108))
                if f == count // 2:
                    canvas.save(root / f'{args.output_name}-{i:02d}.jpg', quality=94)
                if encoder:
                    encoder.stdin.write(canvas.tobytes())
            print(f'{i + 1}/{len(beats)} {beat["mode"]}', flush=True)
    finally:
        renderer.close()
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed')
    if not args.stills:
        print(root / (args.output_name + '.mp4'), flush=True)


if __name__ == '__main__':
    main()
