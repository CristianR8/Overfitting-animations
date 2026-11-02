#!/usr/bin/env python3
"""CVAIL intro. Run with Python + bpy, or blender -b --python animate.py -- ..."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bpy
from mathutils import Vector
import numpy as np

ROOT = Path(__file__).resolve().parent


def arguments():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'config.json')
    p.add_argument('--output', type=Path, default=ROOT / 'renders')
    p.add_argument('--preset', choices=['draft', 'preview', 'full', '4k'], default='preview')
    p.add_argument('--stills', action='store_true', help='Render five representative frames only')
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--start', type=int, default=1, help='Resume from this frame')
    p.add_argument('--end', type=int)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--engine', choices=['eevee', 'cycles'], default='eevee')
    return p.parse_args(argv)


def linear(hex_color):
    rgb = [int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c < .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb) + (1,)


def material(name, color, flat=False, metallic=0, roughness=.5):
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    m.use_nodes = True
    nodes = m.node_tree.nodes
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    if flat:
        shader = nodes.new('ShaderNodeEmission')
        shader.inputs['Color'].default_value = color
        shader.inputs['Strength'].default_value = 1
    else:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = color
        shader.inputs['Roughness'].default_value = roughness
        shader.inputs['Metallic'].default_value = metallic
    m.node_tree.links.new(shader.outputs[0], out.inputs['Surface'])
    return m


def key(obj, path, value, second, fps):
    setattr(obj, path, value)
    obj.keyframe_insert(data_path=path, frame=round(second * fps) + 1)


def window(obj, start, end, fps, scale=1, fade=.35):
    """All visibility is ordinary keyframed scale; no Python handlers in .blend."""
    full = (scale,) * 3
    tiny = (.000001,) * 3
    if start > 0:
        key(obj, 'scale', tiny, max(0, start - .05), fps)
        key(obj, 'scale', full, start + fade, fps)
    else:
        key(obj, 'scale', full, 0, fps)
    if end is not None:
        key(obj, 'scale', full, end - fade, fps)
        key(obj, 'scale', tiny, end, fps)


def empty(name, location=(0, 0, 0)):
    ob = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(ob)
    ob.location = location
    return ob


def line(name, points, mat, radius=.012, parent=None, reveal=None, fps=24):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    ob = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(ob)
    curve.materials.append(mat)
    ob.parent = parent
    if reveal:
        key(curve, 'bevel_factor_end', 0, reveal[0], fps)
        key(curve, 'bevel_factor_end', 1, reveal[1], fps)
    return ob


def text(name, body, x, y, size, mat, font=None, align='LEFT', parent=None, z=6):
    curve = bpy.data.curves.new(name, 'FONT')
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.space_character = 1.06
    curve.space_line = 1.2
    curve.resolution_u = 8
    if font:
        curve.font = font
    ob = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(ob)
    ob.location = (x, y, z)
    ob.parent = parent
    curve.materials.append(mat)
    return ob


def wrap(body, width=35):
    import textwrap
    return '\n'.join(textwrap.wrap(body, width=width))


def prepare_fonts(output):
    """Bake static fonts from Noto's variable font, or use Blender's bundled font."""
    normal = bold = None
    try:
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        source = Path('/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf')
        if source.exists():
            for weight, name in [(400, 'Regular'), (700, 'Bold')]:
                path = output / f'NotoSans-{name}.ttf'
                if not path.exists():
                    font = instantiateVariableFont(TTFont(str(source)), {'wght': weight})
                    font.save(path)
                loaded = bpy.data.fonts.load(str(path))
                if weight == 400:
                    normal = loaded
                else:
                    bold = loaded
    except ImportError:
        pass
    return normal, bold


def dataset(seed):
    rng = np.random.default_rng(seed)
    train = rng.uniform([-2.8, -2.1], [2.8, 2.1], (30, 2))
    test = rng.uniform([-2.8, -2.1], [2.8, 2.1], (18, 2))

    def truth(p):
        x, y = p[:, 0], p[:, 1]
        return .45 * np.sin(x * .9) + .25 * np.cos(y * 1.1) + .08 * x * y

    def features(p):
        x, y = p[:, 0], p[:, 1]
        return np.column_stack([np.ones(len(p)), x, y, x * y, x*x, y*y])

    # Same data-generating process and independent noise for both splits.
    train_y = truth(train) + rng.normal(0, .36, len(train))
    test_y = truth(test) + rng.normal(0, .36, len(test))
    beta = np.linalg.lstsq(features(train), train_y, rcond=None)[0]

    def smooth(p):
        return features(p) @ beta

    def kernel(p):
        distance = np.sum((p[:, None, :] - train[None, :, :]) ** 2, axis=2)
        return np.exp(-distance / (2 * .52 ** 2))

    weights = np.linalg.solve(kernel(train) + np.eye(len(train)) * 1e-6, train_y - smooth(train))

    def overfit(p):
        return smooth(p) + kernel(p) @ weights

    alphas = np.linspace(0, 1, 100)
    errors = []
    for a in alphas:
        pred_train = smooth(train) + a * (overfit(train) - smooth(train))
        pred_test = smooth(test) + a * (overfit(test) - smooth(test))
        errors.append([float(np.mean((pred_train-train_y)**2)), float(np.mean((pred_test-test_y)**2))])
    return train, train_y, test, test_y, smooth, overfit, np.array(errors)


def build(args, cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.file_preview_type = 'NONE'
    scene = bpy.context.scene
    fps = cfg['fps']
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = round(cfg['seconds'] * fps)
    scene.render.engine = 'BLENDER_EEVEE' if args.engine == 'eevee' else 'CYCLES'
    scene.cycles.device = 'CPU'
    presets = {'draft': (640, 360, 4), 'preview': (1280, 720, 8),
               'full': (1920, 1080, 32), '4k': (3840, 2160, 64)}
    w, h, samples = presets[args.preset]
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    scene.eevee.taa_render_samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 3
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 1
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = args.threads
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.film_transparent = False
    world = bpy.data.worlds.new('Ivory studio')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (.74, .74, .74, 1)
    world.node_tree.nodes['Background'].inputs[1].default_value = .6
    scene.world = world

    green = linear(cfg['green'])
    gold = linear(cfg['gold'])
    ivory = linear(cfg['ivory'])
    ink = material('Brand / forest green', green, flat=True)
    accent = material('Brand / champagne gold', gold, flat=True)
    muted = material('Brand / quiet ink', linear('69766C'), flat=True)
    gridmat = material('Stage / grid', linear('D3D8CA'), flat=True)
    surface_mat = material('Model / green ceramic', green, roughness=.42)
    trainmat = material('Training / ivory porcelain', linear('F3EDDD'), roughness=.24)
    testmat = material('Validation / gold', gold, metallic=.35, roughness=.26)
    bgmat = material('Background / ivory', ivory, flat=True)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -7))
    bpy.context.object.name = 'Infinite ivory background'
    bpy.context.object.data.materials.append(bgmat)

    bpy.ops.object.camera_add(location=(0, 0, 30))
    cam = bpy.context.object
    cam.name = 'Editorial orthographic camera'
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 17.78
    cam.data.lens = 50
    scene.camera = cam
    for name, loc, energy, size in [('Softbox key', (-5, 2, 12), 1600, 8),
                                    ('Gold rim', (3, -1, 8), 900, 6)]:
        bpy.ops.object.light_add(type='AREA', location=loc)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.shape = 'DISK'
        light.data.size = size
        light.rotation_euler = (Vector((-3, 0, 0))-light.location).to_track_quat('-Z', 'Y').to_euler()

    normal, bold = prepare_fonts(args.output)
    text('Series wordmark', cfg['series'], -7.9, 4.1, .25, ink, bold)
    text('Edition marker', 'RESEARCH IN MOTION / 01', 7.9, 4.1, .18, muted, normal, align='RIGHT')
    line('Top rule', [(-7.9, 3.75, 6), (7.9, 3.75, 6)], gridmat, .008)
    text('Footer', 'CVAIL', -7.9, -4.4, .2, ink, bold)
    text('Footer right', 'TALK SERIES', 7.9, -4.4, .17, muted, normal, align='RIGHT')

    stage = empty('01 - 3D learning experiment', (-3.7, -.85, 0))
    stage.rotation_euler = (math.radians(54), math.radians(-4), math.radians(-18))
    window(stage, 0, 12.8, fps, scale=1.0, fade=.65)
    for x in np.arange(-3.2, 3.21, .4):
        line('Ground grid X', [(x, -2.5, -.95), (x, 2.5, -.95)], gridmat, .006, stage)
    for y in np.arange(-2.5, 2.51, .4):
        line('Ground grid Y', [(-3.2, y, -.95), (3.2, y, -.95)], gridmat, .006, stage)

    train, train_y, test, test_y, smooth, overfit, errors = dataset(cfg['seed'])
    nx, ny = 85, 65
    xy = np.array([(x, y) for y in np.linspace(-2.4, 2.4, ny) for x in np.linspace(-3.1, 3.1, nx)])
    verts = [(x, y, -.8) for x, y in xy]
    faces = [(j*nx+i, j*nx+i+1, (j+1)*nx+i+1, (j+1)*nx+i)
             for j in range(ny-1) for i in range(nx-1)]
    mesh = bpy.data.meshes.new('Regression surface mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    ob = bpy.data.objects.new('The model - editable shape keys', mesh)
    bpy.context.collection.objects.link(ob)
    ob.parent = stage
    mesh.materials.append(surface_mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    ob.shape_key_add(name='Basis - underfit plane')
    trend = ob.shape_key_add(name='Learn the smooth trend')
    noise = ob.shape_key_add(name='Memorize the training noise')
    for i, (z, residual) in enumerate(zip(smooth(xy), overfit(xy)-smooth(xy))):
        trend.data[i].co.z = float(z)
        noise.data[i].co.z = -.8 + float(residual)
    # Relative shape keys add: plane + trend delta + noise delta.
    key(trend, 'value', 0, 0, fps)
    key(trend, 'value', 1, 3.5, fps)
    key(noise, 'value', 0, 4.4, fps)
    key(noise, 'value', 1, 7.9, fps)
    solid = ob.modifiers.new('Thin sculptural edge', 'SOLIDIFY')
    solid.thickness = .035

    for category, points, heights, mat, start in [
        ('Training', train, train_y, trainmat, .15),
        ('Unseen', test, test_y, testmat, 8.1)]:
        group = empty(category + ' observations')
        group.parent = stage
        for i, ((x, y), z) in enumerate(zip(points, heights)):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=.075,
                                                location=(float(x), float(y), float(z)+.045))
            sphere = bpy.context.object
            sphere.name = f'{category} observation {i+1:02d}'
            sphere.parent = group
            sphere.data.materials.append(mat)
            for poly in sphere.data.polygons:
                poly.use_smooth = True
            window(sphere, start + i * .025, None, fps, fade=.4)
        if category == 'Unseen':
            for i, ((x, y), z, pred) in enumerate(zip(points, heights, overfit(points))):
                line(f'Unseen residual {i+1:02d}',
                     [(float(x), float(y), float(pred)+.025), (float(x), float(y), float(z)+.045)],
                     accent, .012, group, reveal=(8.7+i*.02, 9.35+i*.02), fps=fps)

    for i, (start, end) in enumerate([(0, 4.5), (4.55, 8.15), (8.2, 12.8)]):
        block = empty(f'Chapter {i+1} typography')
        label, title, desc = cfg['chapters'][i]
        text('Chapter number', label, 1.1, 2.3, .19, accent, bold, parent=block)
        text('Chapter title', wrap(title, 19), 1.1, 1.25, .59, ink, bold, parent=block)
        text('Chapter explanation', wrap(desc, 34), 1.1, -.55, .25, muted, normal, parent=block)
        text('Training legend', 'o  '+cfg['training_label'], 1.1, -2.3, .18, ink, bold, parent=block)
        if i == 2:
            text('Validation legend', 'o  '+cfg['validation_label'], 1.1, -2.72, .18, accent, bold, parent=block)
            train_error, val_error = errors[-1]
            text('Measured training error', f"{cfg['train_error_label']}  {train_error:.3f}", 1.1, -3.3, .18, muted, normal, parent=block)
            text('Measured held-out error', f"{cfg['validation_error_label']}  {val_error:.3f}", 4.05, -3.3, .18, accent, normal, parent=block)
        window(block, start, end, fps, fade=.25)

    closing = empty('02 - Talk series closing card')
    window(closing, 12.65, None, fps, fade=.8)
    text('Closing kicker', 'A CVAIL TALK SERIES', 0, 2.25, .22, accent, bold, align='CENTER', parent=closing)
    text('Closing title', cfg['title'], 0, .95, .95, ink, bold, align='CENTER', parent=closing)
    if cfg['guest']:
        text('Guest name', cfg['guest'], 0, -.08, .52, ink, normal, align='CENTER', parent=closing)
    else:
        text('Closing ellipsis', '...', 0, -.08, .58, accent, bold, align='CENTER', parent=closing)
    text('Closing statement', cfg['tagline'], 0, -1.05, .27, muted, normal, align='CENTER', parent=closing)
    max_error = float(errors.max()) * 1.12
    for index, label, mat in [(0, 'Training', ink), (1, 'Validation', accent)]:
        points = [(-6.6 + 13.2*j/(len(errors)-1), -3.65 + 1.65*float(e[index])/max_error, 5.5)
                  for j, e in enumerate(errors)]
        if index == 0:
            line('Measured training MSE curve', points, mat, .025, closing,
                 reveal=(12.85, 14.5), fps=fps)
        else:
            for j in range(0, len(points)-2, 5):
                line('Measured validation MSE dash', points[j:j+3], mat, .025, closing,
                     reveal=(12.85+j*.014, 13.3+j*.014), fps=fps)
    text('Curve legend', 'ERROR  /  MODEL COMPLEXITY', -6.6, -1.88, .15, muted, normal, parent=closing)
    text('Training curve label', cfg['train_error_label'], 6.65, -3.88, .15, ink, bold,
         align='RIGHT', parent=closing)
    text('Validation curve label', cfg['validation_error_label'], 6.65, -1.88, .15, accent, bold,
         align='RIGHT', parent=closing)

    for sec, name in [(0, 'Learn the trend'), (4.5, 'Fit the noise'), (8.2, 'Reveal unseen data'),
                      (12.65, 'Talk series closing')]:
        scene.timeline_markers.new(name, frame=round(sec*fps)+1)
    scene['CVAIL / README'] = 'All motion is keyframed. Edit config.json and rebuild for text, colors and speaker.'
    scene['CVAIL / Model'] = 'Quadratic least squares + Gaussian RBF interpolation of residuals. Independent same-distribution validation split.'
    scene['CVAIL / Train MSE'] = float(errors[-1, 0])
    scene['CVAIL / Validation MSE'] = float(errors[-1, 1])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps({
        'seed': cfg['seed'], 'n_train': len(train), 'n_validation': len(test),
        'smooth_train_mse': float(errors[0, 0]), 'smooth_validation_mse': float(errors[0, 1]),
        'overfit_train_mse': float(errors[-1, 0]), 'overfit_validation_mse': float(errors[-1, 1]),
        'note': 'Synthetic regression; validation data never used to fit or modify the surface.'
    }, indent=2))
    scene.frame_set(1)
    # Packed fonts make the .blend portable.
    bpy.ops.file.pack_all()
    # Opening the .blend and choosing Render Animation also produces an MP4.
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'VIDEO'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'HIGH'
    scene.render.filepath = '//CVAIL-overfitting-native.mp4'
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'CVAIL-overfitting.blend'))
    if hasattr(scene.render.image_settings, 'media_type'):
        scene.render.image_settings.media_type = 'IMAGE'
    scene.render.image_settings.file_format = 'PNG'
    print('CVAIL_BUILT', json.dumps({'size': [w, h], 'frames': scene.frame_end,
                                    'train_mse': float(errors[-1,0]), 'validation_mse': float(errors[-1,1])}), flush=True)
    return scene


def render(args, scene, cfg):
    frames_dir = args.output / 'frames'
    frames_dir.mkdir(parents=True, exist_ok=True)
    if args.stills:
        frames = [round(t*cfg['fps'])+1 for t in [2.8, 6.5, 9.8, 13.8, 16.5]]
    else:
        frames = range(args.start, min(args.end or scene.frame_end, scene.frame_end)+1)
    began = time.monotonic()
    cached_holds = {}
    for frame in frames:
        path = frames_dir / f'{frame:05d}.png'
        if path.exists():
            continue
        # These intervals have no changing keyframes. Reuse the identical image
        # without lowering the video's frame rate or changing the editable scene.
        second = (frame - 1) / cfg['fps']
        hold = 'reveal' if 9.75 <= second < 12.14 else ('closing' if second >= 14.7 else None)
        if hold and hold in cached_holds and not args.stills:
            shutil.copyfile(cached_holds[hold], path)
            continue
        scene.frame_set(frame)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        if hold:
            cached_holds[hold] = path
        print(f'CVAIL_FRAME {frame}/{scene.frame_end} elapsed={time.monotonic()-began:.1f}s', flush=True)
    if args.stills:
        return
    # Require every frame before encoding: never silently deliver a truncated movie.
    missing = [f for f in range(1, scene.frame_end+1) if not (frames_dir / f'{f:05d}.png').exists()]
    if missing:
        print(f'Partial render: {len(missing)} frames remain. Rerun the same command to resume.', flush=True)
        return
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('PNG frames are ready. Install FFmpeg to encode the MP4.')
    target = args.output / 'CVAIL-overfitting.mp4'
    encoders = subprocess.run([ffmpeg, '-hide_banner', '-encoders'],
                              capture_output=True, text=True, check=True).stdout
    if 'libx264' in encoders:
        codec_options = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18']
    elif 'libopenh264' in encoders:
        # Fedora's FFmpeg may provide OpenH264 rather than libx264.
        codec_options = ['-c:v', 'libopenh264', '-b:v', '8M']
    else:
        raise RuntimeError('PNG frames are ready. FFmpeg needs libx264 or libopenh264 for H.264.')
    subprocess.run([ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-threads', str(args.threads),
                    '-framerate', str(cfg['fps']), '-start_number', '1', '-i', str(frames_dir/'%05d.png'),
                    *codec_options, '-threads', str(args.threads), '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', '-an', str(target)], check=True)
    print(f'CVAIL_MP4 {target}', flush=True)


if __name__ == '__main__':
    args = arguments()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    if cfg['seconds'] < 18:
        raise ValueError('This storyboard requires seconds >= 18. Change fps or preset for a faster render.')
    scene = build(args, cfg)
    if not args.build_only:
        render(args, scene, cfg)
