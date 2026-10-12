"""Edit saved camera evidence and recorded MuJoCo motion into a short dial demo.

Identification, planning and verification use saved side/wrist observations.
Fixed crops preserve those cameras' perspectives; a presentation camera is
used only for motion. Overlays illustrate recorded
decisions and public measurements; they are not inputs to the policy. Motion
plays at 1x; deliberation and tool latency are shortened.
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
from PIL import Image, ImageDraw, ImageFont

from world_use.bodies.mujoco_render import CameraRenderer
from world_use.cameras import View

W, H, FPS = 1920, 1080, 30
PAPER, INK, MUTED = '#f8f7f3', '#263331', '#7b827e'
TEAL, MINT, BLUE = '#257e6c', '#a6f1db', '#466aa5'


@lru_cache
def font(size, mono=False):
    return ImageFont.truetype('/System/Library/Fonts/'+('SFNSMono.ttf' if mono else 'SFNS.ttf'), size)


@lru_cache
def grade():
    ramp = np.arange(256)/255
    return np.concatenate([np.round(np.clip(ramp*g, 0, 1)**.91*255)
                           for g in (.985, 1.025, 1.10)]).astype('uint8').tolist()


def ease(t):
    t = np.clip(t, 0, 1)
    return float(t*t*(3-2*t))


def transform(center, width, angle=0, target=(960, 540)):
    a = np.radians(angle)
    rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])*W/width
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = target-rotation@center
    return matrix


def warp(image, matrix):
    inverse = np.linalg.inv(matrix)
    return image.transform((W, H), Image.Transform.AFFINE, tuple(inverse[:2].flatten()),
                           Image.Resampling.BICUBIC)


class Film:
    def __init__(self, root):
        self.root = root
        self.calls = json.loads((root/'calls.json').read_text())
        self.notes = {n['stage']: n for n in json.loads((root/'decisions.json').read_text())}
        self.audit = json.loads((root/'physics-audit.json').read_text())
        assert self.audit['passed']
        self.replay = np.load(root/'replay.npz')
        self.model = mj.MjModel.from_binary_path(str(root/'scene.mjb'))
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = W, H
        self.model.vis.quality.offsamples = 8
        self.data = mj.MjData(self.model)
        self.data.qpos[:] = self.replay['qpos'][0]
        mj.mj_forward(self.model, self.data)
        self.base = np.eye(4)
        self.base[:3, :3] = self.data.body('robot0_link0').xmat.reshape(3, 3)
        self.base[:3, 3] = self.data.body('robot0_link0').xpos
        self.renderer = CameraRenderer()
        self.cache = {}
        self.center = np.asarray(self.notes['refine']['center'])
        self.prepare_opening()
        self.prepare_verification()
        self.edit = []
        def add(mode, duration, chapter, owner, command, reply, call=None):
            self.edit.append(dict(mode=mode, duration=duration, chapter=chapter, owner=owner,
                                  command=command, reply=reply, call=call))
        add('intro', 1.6, 'Six controls. One request.', '', '', '')
        add('identify', 3, 'Identify the control', 'LLM · reads the image',
            'Read labels + symbols', 'Choose temperature')
        add('cloud', 1.8, 'Locate the handle', 'world-use · returned evidence',
            'select_target(frame, box)', 'EdgeTAM + RGB-D')
        add('measure', 2.2, 'Measure the marks', 'world-use · returned evidence',
            'measure_pixels(frame, point)', '3D center + scale marks')
        add('angle', 1.8, 'Calculate the turn', 'LLM · task Python',
            '180°C → 356°F', '43° planned rotation')
        for index, mode, chapter, request, reply in [
            (11, 'approach', 'Approach', 'move_to(clear_pose)', 'Align with the handle'),
            (12, 'approach', 'Reach the handle', 'move_to(grasp_pose)', 'Measured center + axis'),
            (13, 'grip', 'Grip', 'grip(expect_mm=[7, 18])', 'Contact at 13 mm'),
            (14, 'turn', 'Turn', 'move_to(center, jaws=target)', 'Rotate about the dial axis'),
            (15, 'withdraw', 'Release and withdraw', 'run([open, back, clear])', 'Leave the dial in place'),
        ]:
            c = self.calls[index]
            add(mode, c['end']-c['start'], chapter, 'world-use · executing', request, reply, index)
        add('observe', 1.1, 'Observe after release', 'LLM → world-use',
            'camera_frame("wrist")', 'Fresh image after release')
        add('verify', 1.5, 'Read the pointer', 'world-use → LLM',
            'measure_pixels(frame, point)', f'Pointer ≈{self.reading_f:.0f}°F')
        add('judge', 2, 'Verify the result', 'LLM · checks fresh evidence',
            f'{self.requested_f:.0f}°F target → ≈{self.reading_f:.0f}°F read',
            f'≈{abs(self.error_f):.0f}°F error · accept')

    def point(self, index):
        return np.asarray(self.calls[index]['result']['surface_center'])

    def projection(self, points):
        uv = self.view.project(points)[0]
        return uv@self.matrix[:2, :2].T+self.matrix[:2, 2]

    def prepare_verification(self):
        """Rebuild the recorded visual check from its fresh public measurements."""
        self.check_center = self.point(20)
        self.mark_angles = np.array([np.arctan2(*(self.point(i)-self.check_center)[[2, 1]])
                                     for i in (21, 22)])
        pointer_angle = np.arctan2(*(self.point(19)-self.check_center)[[2, 1]])
        self.reading_f = float(350+50*(pointer_angle-self.mark_angles[0])/np.diff(self.mark_angles)[0])
        self.requested_f = self.notes['verify']['requested_f']
        self.error_f = self.reading_f-self.requested_f
        self.tolerance_f = self.audit['visual_tolerance_f']
        assert np.isclose(self.reading_f, self.notes['verify']['indicated_f'])
        assert self.notes['verify']['passed'] and abs(self.error_f)<self.tolerance_f
        frame = self.calls[18]['result']['frame']
        frame = frame['id'] if isinstance(frame, dict) else frame
        assert frame==self.notes['verify']['frame']
        assert self.calls[18]['start']>self.calls[15]['end']
        assert all(self.calls[i]['arguments']['frame']==frame for i in range(19, 23))

    def scale_point(self, fahrenheit, radius):
        angle = self.mark_angles[0]+(fahrenheit-350)/50*np.diff(self.mark_angles)[0]
        point = self.check_center.copy()
        point[0] = self.point(19)[0]
        point[1:] += radius*np.array([np.cos(angle), np.sin(angle)])
        return point

    def verification(self, layer, mode, local):
        """Reveal the evidence first, then the comparison and recorded acceptance."""
        if mode=='verify':
            marks = [(21, '350°F', [-260, -145]), (22, '400°F', [240, 155]),
                     (19, f'Read ≈{self.reading_f:.0f}°F', [250, 10])]
            for order, (index, label, offset) in enumerate(marks):
                detail = Image.new('RGBA', (W, H))
                self.tag(detail, self.projection([self.point(index)])[0], label, np.array(offset), small=order<2)
                opacity = ease((local-order*.27)/.28)
                detail.putalpha(detail.getchannel('A').point(lambda v, opacity=opacity: int(v*opacity)))
                layer.alpha_composite(detail)
            return
        d = ImageDraw.Draw(layer)
        values = np.linspace(self.requested_f-self.tolerance_f,
                             self.requested_f-self.tolerance_f+2*self.tolerance_f*ease(local/.45), 50)
        band = self.projection([self.scale_point(t, .027) for t in values])
        d.line(list(map(tuple, band)), fill=PAPER, width=18, joint='curve')
        d.line(list(map(tuple, band)), fill=MINT, width=11, joint='curve')
        tick = self.projection([self.scale_point(self.requested_f, radius) for radius in (.024, .031)])
        d.line(list(map(tuple, tick)), fill=PAPER, width=8)
        d.line(list(map(tuple, tick)), fill=BLUE, width=3)
        self.tag(layer, tick[-1], f'Target {self.requested_f:.0f}°F', np.array([-375, -145]), BLUE)
        self.tag(layer, self.projection([self.point(19)])[0],
                 f'Read ≈{self.reading_f:.0f}°F', np.array([250, 10]))

    @staticmethod
    def view_label(mode):
        if mode in ('observe', 'verify', 'judge'):
            return 'Wrist camera', 'Model input · after release'
        if mode in ('measure', 'angle'):
            return 'Wrist camera', 'Model input · upright crop'
        if mode in ('intro', 'identify', 'cloud'):
            return 'Side camera', 'Model input'
        return 'Side view', 'Simulation replay · 1×'  # noqa: RUF001

    def saved(self, final=False, side=False):
        frame = (self.notes['verify']['frame'] if final else
                 self.calls[2 if side else 3]['result']['frame'])
        if isinstance(frame, dict):
            frame = frame['id']
        key = ('saved', frame)
        if key not in self.cache:
            im = Image.open(self.root/'frames'/f'{frame}.png').convert('RGB')
            view = View.from_dict(json.loads((self.root/'frames'/f'{frame}.json').read_text()))
            if side:
                matrix = transform(np.array([im.width/2, im.height/2]), im.width)
            else:
                c = self.point(20 if final else 9)
                uv = view.project([c, c+[0, 0, .03]])[0]
                v = uv[1]-uv[0]
                angle = np.degrees(np.arctan2(v[1], v[0]))+90
                matrix = transform(uv[0], 760 if final else 800, angle, np.array([1010, 550]))
            im = im.point(grade())
            self.cache[key] = warp(im, matrix), view, matrix
        pic, self.view, self.matrix = self.cache[key]
        return pic.copy()

    def prepare_opening(self):
        """Keep presentation views separate from the two recorded policy cameras."""
        self.action_view = View.look_at([2.30, -1.18, 1.28], [3.20, -.73, .89], fov_deg=43, size=(W, H))
        # Editorial anchors recovered from the original side RGB-D, not scene labels.
        frame = self.calls[2]['result']['frame']
        frame = frame['id'] if isinstance(frame, dict) else frame
        view = View.from_dict(json.loads((self.root/'frames'/f'{frame}.json').read_text()))
        depth = np.load(self.root/'frames'/f'{frame}.npz')['depth']
        pixels = np.array([[264, 550], [475, 530], [735, 510]])
        self.controls = view.unproject(pixels, depth[pixels[:, 1], pixels[:, 0]])
        assert self.calls[4]['arguments']['frame']==frame
        assert self.calls[4]['arguments']['box']==self.notes['interpret']['box']

    def opening(self, mode, local):
        """Hold the side input fixed while its identification overlays animate."""
        if mode in ('intro', 'identify', 'cloud'):
            return self.saved(side=True)
        if mode=='measure' and local<.35:
            before = self.saved(side=True)
            after = self.saved()
            return Image.blend(before, after, ease(local/.35))
        return self.saved()

    def selection_box(self, layer, progress=1):
        """The actual pixel box submitted to select_target, in its source image."""
        x1, y1, x2, y2 = self.calls[4]['arguments']['box']
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
        corners = corners@self.matrix[:2, :2].T+self.matrix[:2, 2]
        d = ImageDraw.Draw(layer)
        path = [corners[0]]
        for i in range(4):
            t = np.clip(progress*4-i, 0, 1)
            path.append(corners[i]*(1-t)+corners[i+1]*t)
            if t<1:
                break
        d.line(list(map(tuple, path)), fill=PAPER, width=7, joint='curve')
        d.line(list(map(tuple, path)), fill=TEAL, width=3, joint='curve')
        return np.mean(corners[:4], axis=0)

    def control_highlights(self, overlay, local):
        if local < .8:
            centers, label, color = self.controls[:2], 'Burners', MUTED
            progress = local
            alpha = ease(local/.18)*(1-ease((local-.6)/.2))
        elif local < 1.65:
            centers, label, color = self.controls[2:3], 'Timer · 0–60', MUTED  # noqa: RUF001
            progress = local-.8
            alpha = ease(progress/.18)*(1-ease((local-1.45)/.2))
        else:
            centers, label, color = [], 'Temperature · °F', TEAL
            progress = local-1.65
            alpha = ease(progress/.2)*(1-ease((local-2.8)/.2))
        layer = Image.new('RGBA', (W, H))
        draw = ImageDraw.Draw(layer)
        for center in centers:
            a = np.linspace(0, 2*np.pi*ease(progress/.4), 100)
            ring = np.asarray(center)+np.column_stack([np.zeros(len(a)), .033*np.sin(a), .033*np.cos(a)])
            uv = self.projection(ring)
            draw.line(list(map(tuple, uv)), fill=PAPER, width=7, joint='curve')
            draw.line(list(map(tuple, uv)), fill=color, width=3, joint='curve')
        anchor = (self.projection([np.mean(centers, axis=0)])[0] if len(centers) else
                  self.selection_box(layer, ease(progress/.4)))
        self.tag(layer, anchor, label, np.array([150, 145]), color)
        layer.putalpha(layer.getchannel('A').point(lambda v: int(v*alpha)))
        overlay.alpha_composite(layer)

    def render_view(self, stamp, view):
        index = int(np.clip(np.searchsorted(self.replay['time'], stamp), 0, len(self.replay['time'])-1))
        self.data.qpos[:] = self.replay['qpos'][index]
        mj.mj_forward(self.model, self.data)
        self.view = replace(view, T=np.linalg.inv(self.base)@view.T)
        self.matrix = np.eye(3)
        key = ('render', index, tuple(view.T.flatten()), view.fx, view.fy, view.cx, view.cy)
        if key not in self.cache:
            for k in list(self.cache):
                if k[0]=='render':
                    del self.cache[k]
            self.cache[key] = self.renderer.render(self.model, self.data, view).point(grade())
        return self.cache[key].copy()

    def motion(self, stamp):
        return self.render_view(stamp, self.action_view)

    def tag(self, layer, anchor, text, offset, color=TEAL, small=False):
        d = ImageDraw.Draw(layer)
        x, y = np.asarray(anchor)+offset
        size = 24 if small else 30
        width = d.textlength(text, font=font(size))+36
        height = 52 if small else 62
        x, y = float(np.clip(x, 32, W-width-32)), float(np.clip(y, 140, 910-height))
        end = (x if anchor[0]<x else x+width if anchor[0]>x+width else anchor[0], y+height/2)
        d.line([tuple(anchor), end], fill=PAPER, width=5)
        d.line([tuple(anchor), end], fill=color, width=2)
        ax, ay = anchor
        d.ellipse((ax-5, ay-5, ax+5, ay+5), fill=color, outline=PAPER, width=2)
        d.rounded_rectangle((x, y, x+width, y+height), radius=14, fill=PAPER)
        d.text((x+18, y+13), text, font=font(size), fill=color)

    def arc(self, layer, progress=1, radius=.045):
        plan = self.notes['refine']
        start, end = np.radians([plan['start_deg'], plan['target_deg']])
        angles = np.linspace(start, start+(end-start)*max(.025, progress), 70)
        c = self.center.copy()
        c[0] = .482
        pts = c+np.column_stack([np.zeros(len(angles)), radius*np.cos(angles), radius*np.sin(angles)])
        uv = self.projection(pts)
        d = ImageDraw.Draw(layer)
        origin = self.projection([c])[0]
        for endpoint in (uv[0], uv[-1]):
            for lo in np.arange(.12, 1, .06):
                a = origin+(endpoint-origin)*lo
                b = origin+(endpoint-origin)*min(lo+.027, 1)
                d.line([tuple(a), tuple(b)], fill=PAPER, width=5)
                d.line([tuple(a), tuple(b)], fill=BLUE, width=2)
        d.ellipse((origin[0]-5, origin[1]-5, origin[0]+5, origin[1]+5), fill=BLUE, outline=PAPER, width=2)
        d.line(list(map(tuple, uv)), fill=PAPER, width=11, joint='curve')
        d.line(list(map(tuple, uv)), fill=BLUE, width=5, joint='curve')
        v = uv[-1]-uv[-3]
        v /= np.linalg.norm(v)
        n = np.array([-v[1], v[0]])
        tip = uv[-1]
        d.polygon([tuple(tip+5*v), tuple(tip-20*v+12*n), tuple(tip-20*v-12*n)], fill=PAPER)
        d.polygon([tuple(tip+2*v), tuple(tip-16*v+8*n), tuple(tip-16*v-8*n)], fill=BLUE)
        self.tag(layer, uv[len(uv)//2], '43° planned', np.array([110, -145]), BLUE)

    def draw(self, beat, local):
        mode = beat['mode']
        if beat['call'] is not None:
            c = self.calls[beat['call']]
            previous = self.saved() if beat['call']==11 and local<.35 else None
            pic = self.motion(min(c['end'], c['start']+local))
            if previous is not None:
                pic = Image.blend(previous, pic, ease(local/.35))
        elif mode in ('observe', 'verify', 'judge'):
            previous = self.motion(self.calls[15]['end']) if mode=='observe' and local < .35 else None
            pic = self.saved(final=True)
            if previous is not None:
                pic = Image.blend(previous, pic, ease(local/.35))
        else:
            pic = self.opening(mode, local)
        layer = Image.new('RGBA', (W, H))
        d = ImageDraw.Draw(layer)
        if mode=='identify':
            self.control_highlights(layer, local)
        elif mode=='cloud':
            points = np.load(self.root/'flight/perception'/f'{self.calls[4]["result"]["id"]}.npz')['points']
            bounds = np.array(self.calls[4]['result']['visible_bounds'])
            points = points[np.all((points>=bounds[0])&(points<=bounds[1]), axis=1)]
            uv = self.projection(points)
            count = int(len(uv)*ease(local/.7))
            for x, y in uv[:count:2]:
                d.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=(*bytes.fromhex('a6f1db'), 160))
            anchor = self.selection_box(layer)
            self.tag(layer, anchor, 'Visible surface → 3D', np.array([160, 145]))
        elif mode=='measure':
            for order, (i, name, offset) in enumerate([(9, 'Center', [-280, 120]),
                                                       (6, '350°F', [-250, -160]), (7, '400°F', [155, 50])]):
                detail = Image.new('RGBA', (W, H))
                self.tag(detail, self.projection([self.point(i)])[0], name, np.array(offset), small=True)
                opacity = ease((local-.35-order*.3)/.3)
                detail.putalpha(detail.getchannel('A').point(lambda v, opacity=opacity: int(v*opacity)))
                layer.alpha_composite(detail)
        elif mode in ('angle', 'turn'):
            self.arc(layer, ease(local/.75) if mode=='angle' else 1,
                     radius=.035 if mode=='angle' else .065)
        elif mode in ('verify', 'judge'):
            self.verification(layer, mode, local)
        fade_start = .35 if mode=='measure' else 0
        fade_out = 1 if mode=='judge' else 1-ease((local-beat['duration']+.2)/.2)
        opacity = ease((local-fade_start)/.18)*fade_out
        layer.putalpha(layer.getchannel('A').point(lambda v: int(v*opacity)))
        pic = Image.alpha_composite(pic.convert('RGBA'), layer).convert('RGB')
        d = ImageDraw.Draw(pic)
        for y1, y2 in ((24, 96), (944, 1056)):
            d.rounded_rectangle((480, y1, 1440, y2), radius=26, fill=PAPER)
        d.rounded_rectangle((504, 44, 510, 77), radius=3, fill=TEAL)
        d.text((526, 39), 'world-use', font=font(33), fill=INK)
        camera, source = self.view_label(mode)
        if local<.35 and (mode in ('observe', 'measure') or beat['call']==11):
            camera = 'Wrist → side' if beat['call']==11 else 'Side → wrist'
            source = 'View transition'
        d.text((734, 34), camera, font=font(19), fill=TEAL)
        d.text((734, 61), source, font=font(16), fill=MUTED)
        title = beat['chapter']
        if mode=='judge' and local>=.6:
            title = 'Verified'
        d.text((1416-d.textlength(title, font=font(24)), 46), title, font=font(24), fill=MUTED)
        if mode=='intro':
            d.text((514, 976), 'Set the oven to 180°C.', font=font(35), fill=INK)
            d.text((1100, 986), 'Franka Panda · MuJoCo', font=font(21), fill=MUTED)
        else:
            d.text((504, 965), beat['owner'], font=font(17), fill=TEAL)
            d.text((504, 1003), beat['command'], font=font(21, True), fill=INK)
            d.line((1016, 967, 1016, 1033), fill='#d9ddd5')
            caption = ('Feedback' if mode=='grip' else 'Command' if beat['call'] is not None else
                       f'Tolerance ±{self.tolerance_f:g}°F' if mode=='judge' else
                       'Read-back' if mode=='verify' else
                       'Returned observation' if mode=='observe' else
                       'Result' if mode in ('cloud', 'measure') else 'Decision')
            d.text((1040, 965), caption, font=font(17), fill=MUTED)
            reply = beat['reply']
            if mode=='identify':
                reply = 'Burner controls' if local<.8 else 'Oven timer' if local<1.65 else 'Temperature · °F'
            elif mode=='verify' and local<.85:
                reply = 'Fresh pointer + scale marks'
            elif mode=='judge' and local<.6:
                reply = f'≈{abs(self.error_f):.0f}°F error'
            d.text((1040, 1003), reply, font=font(23), fill=TEAL if mode=='judge' and local>=.6 else INK)
            assert d.textlength(beat['command'], font=font(21, True)) < 488
            assert d.textlength(reply, font=font(23)) < 376
        return pic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--stills', action='store_true')
    parser.add_argument('--name', default='kitchen-dial-v7')
    args = parser.parse_args()
    film = Film(args.recording)
    (args.recording/f'{args.name}-edit.json').write_text(json.dumps(film.edit, indent=2)+'\n')
    encoder = None
    if not args.stills:
        encoder = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
            '-video_size', f'{W}x{H}', '-framerate', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(args.recording/f'{args.name}.mp4')], stdin=subprocess.PIPE)
    try:
        for i, beat in enumerate(film.edit):
            count = max(1, round(beat['duration']*FPS))
            for f in ([count//2] if args.stills else range(count)):
                pic = film.draw(beat, f/FPS)
                if f == count//2:
                    pic.save(args.recording/f'{args.name}-{i:02d}.jpg', quality=95)
                if encoder:
                    encoder.stdin.write(pic.tobytes())
            print(f'{i+1}/{len(film.edit)} {beat["mode"]}', flush=True)
    finally:
        film.renderer.close()
        if encoder:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError('ffmpeg failed')


if __name__ == '__main__':
    main()
