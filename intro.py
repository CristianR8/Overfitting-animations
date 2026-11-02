#!/usr/bin/env python3
"""The reference intro translated to 3D: two strokes, one split, one title."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'intro.json')
    p.add_argument('--output', type=Path, default=ROOT/'renders'/'intro-v2')
    p.add_argument('--preset', choices=['draft', 'preview', 'full', '4k'], default='full')
    p.add_argument('--stills', action='store_true')
    p.add_argument('--build-only', action='store_true')
    argv = sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else sys.argv[1:]
    return p.parse_args(argv)


def color(h):
    v = [int(h[i:i+2], 16)/255 for i in [0, 2, 4]]
    return tuple(x/12.92 if x < .04045 else ((x+.055)/1.055)**2.4 for x in v)+(1,)


def mat(name, rgb, flat=False):
    m = bpy.data.materials.new(name)
    m.diffuse_color = rgb
    nodes = m.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    if flat:
        shader = nodes.new('ShaderNodeEmission')
        shader.inputs['Color'].default_value = rgb
    else:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = rgb
        shader.inputs['Roughness'].default_value = .52
        shader.inputs['Metallic'].default_value = 0
        shader.inputs['Specular IOR Level'].default_value = .20
    m.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
    return m


def key(ob, field, value, seconds, fps):
    setattr(ob, field, value)
    ob.keyframe_insert(data_path=field, frame=round(seconds*fps)+1)


def ease(t):
    t = min(1, max(0, t))
    return t*t*(3-2*t)


def sig(x):
    return 1/(1+np.exp(-x))


def path(t, gold=False):
    t = np.asarray(t)
    x = -12+24*t
    if gold:
        y = -2.9+2.35*np.exp(-((t-.405)/.16)**2)-1.12*sig((t-.60)*15)
    else:
        y = -2.05+6.3*sig((t-.40)*16)
    return np.column_stack([x, y, np.full_like(x, .34)])


def resample(gold=False, count=241):
    dense = path(np.linspace(0, 1, 3000), gold)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    targets = np.linspace(0, arc[-1], count)
    return np.column_stack([np.interp(targets, arc, dense[:,i]) for i in range(3)])


def profile(width=.20, depth=.18):
    data = bpy.data.curves.new('Rounded ribbon cross section', 'CURVE')
    data.dimensions = '2D'
    coords = []
    radius = min(width, depth)*.25
    for cx, cy, start in [(width/2-radius, depth/2-radius, 0),
                          (-width/2+radius, depth/2-radius, 90),
                          (-width/2+radius, -depth/2+radius, 180),
                          (width/2-radius, -depth/2+radius, 270)]:
        for angle in np.linspace(start, start+90, 5):
            a = math.radians(angle)
            coords.append((cx+radius*math.cos(a), cy+radius*math.sin(a), 0))
    sp = data.splines.new('POLY')
    sp.points.add(len(coords)-1)
    for v, xyz in zip(sp.points, coords):
        v.co = (*xyz, 1)
    sp.use_cyclic_u = True
    ob = bpy.data.objects.new('Ribbon profile (off camera)', data)
    bpy.context.collection.objects.link(ob)
    ob.location = (100, 100, 100)
    return ob


def ribbon(name, points, material, cross_section, fps, start=0, end=3.6):
    c = bpy.data.curves.new(name, 'CURVE')
    c.dimensions = '3D'
    c.twist_mode = 'Z_UP'
    c.resolution_u = 2
    c.bevel_mode = 'OBJECT'
    c.bevel_object = cross_section
    c.use_fill_caps = True
    sp = c.splines.new('POLY')
    sp.points.add(len(points)-1)
    # Each stroke starts horizontal, then bends into its reference curve.
    base = points.copy()
    base[:,1] = -2.9 if 'Gold' in name else -2.05
    for v, xyz in zip(sp.points, base):
        v.co = (*xyz, 1)
    ob = bpy.data.objects.new(name, c)
    bpy.context.collection.objects.link(ob)
    c.materials.append(material)
    ob.shape_key_add(name='Straight stroke')
    bent = ob.shape_key_add(name='Reference curve')
    for v, xyz in zip(bent.data, points):
        v.co = xyz
    key(bent, 'value', 0, .8, fps)
    key(bent, 'value', 1, 4.2, fps)
    key(c, 'bevel_factor_end', .22 if name.startswith('Green') else 0, start, fps)
    key(c, 'bevel_factor_end', 1, end, fps)
    return ob


def corner(points, material, fps):
    """The original green upper-left fill, now a softly bevelled solid."""
    verts = []
    n = len(points)
    for z in [.05, .25]:
        for top in [False, True]:
            for x, y, _ in points:
                verts.append((x, y+.06 if top else y, z))
    faces = []
    for i in range(n-1):
        faces += [(i, i+1, n+i+1, n+i),
                  (2*n+i, 3*n+i, 3*n+i+1, 2*n+i+1),
                  (i, 2*n+i, 2*n+i+1, i+1),
                  (n+i, n+i+1, 3*n+i+1, 3*n+i)]
    faces += [(0,n,3*n,2*n), (n-1,3*n-1,4*n-1,2*n-1)]
    mesh = bpy.data.meshes.new('Upper-left brand shape')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    bm=bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    ob = bpy.data.objects.new('Green corner / original composition', mesh)
    bpy.context.collection.objects.link(ob)
    mesh.materials.append(material)
    ob.shape_key_add(name='Thin edge')
    full = ob.shape_key_add(name='Fill original green corner')
    for zlayer in [0, 1]:
        for i in range(n):
            full.data[zlayer*2*n+n+i].co.y = 8
    key(full, 'value', 0, 4.25, fps)
    key(full, 'value', 1, 6.5, fps)
    # Before the stroke reaches its final silhouette, keep the fill below the stage.
    key(ob, 'location', (0,0,-2), 0, fps)
    key(ob, 'location', (0,0,-2), 4.2, fps)
    key(ob, 'location', (0,0,0), 4.45, fps)
    bevel = ob.modifiers.new('Soft machined edge', 'BEVEL')
    bevel.width = .055
    bevel.segments = 3
    bevel.limit_method = 'ANGLE'
    ob.modifiers.new('Weighted corner normals', 'WEIGHTED_NORMAL')
    return ob


def arrowhead(name, material):
    mesh=bpy.data.meshes.new(name)
    triangle=[(.12,0),(-.48,.29),(-.48,-.29)]
    verts=[(x,y,z) for z in [-.09,.09] for x,y in triangle]
    mesh.from_pydata(verts,[],[(2,1,0),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5)])
    ob=bpy.data.objects.new(name,mesh)
    bpy.context.collection.objects.link(ob)
    mesh.materials.append(material)
    bevel=ob.modifiers.new('Rounded arrow tip','BEVEL')
    bevel.width=.025
    bevel.segments=3
    return ob


def title(name, body, location, size, material, font_path, fps, start, align='CENTER'):
    c = bpy.data.curves.new(name, 'FONT')
    c.body = body
    c.align_x = align
    c.size = size
    c.extrude = .012
    c.bevel_depth = .004
    c.bevel_resolution = 2
    if font_path.exists():
        c.font = bpy.data.fonts.load(str(font_path))
    ob = bpy.data.objects.new(name, c)
    bpy.context.collection.objects.link(ob)
    c.materials.append(material)
    x,y,z = location
    # Rise through the white stage: a physical reveal, no captions or cards.
    key(ob, 'location', (x,y-.12,-.9), start, fps)
    key(ob, 'location', (x,y,z), start+.85, fps)
    return ob


def build(a, cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.file_preview_type = 'NONE'
    s = bpy.context.scene
    fps = cfg['fps']
    s.render.engine = 'BLENDER_EEVEE'
    w,h,samples = {'draft':(640,360,4), 'preview':(1280,720,8),
                   'full':(1920,1080,8), '4k':(3840,2160,16)}[a.preset]
    s.render.resolution_x=w
    s.render.resolution_y=h
    s.render.resolution_percentage=100
    s.eevee.taa_render_samples=samples
    s.render.fps=fps
    s.frame_start=1
    s.frame_end=round(cfg['seconds']*fps)
    s.view_settings.view_transform='Standard'
    s.view_settings.look='None'
    s.render.image_settings.color_mode='RGB'
    world=bpy.data.worlds.new('White studio')
    world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.8,.8,1)
    world.node_tree.nodes['Background'].inputs[1].default_value=.35
    s.world=world
    green=mat('Forest green satin',color(cfg['green']))
    gold=mat('Champagne gold satin',color(cfg['gold']))
    white=mat('White stage',color(cfg['background']))
    ink=mat('Title green',color(cfg['green']),flat=True)
    accent=mat('Series gold',color(cfg['gold']),flat=True)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0,0,-.52))
    bpy.context.object.name='White cyclorama'
    bpy.context.object.data.materials.append(white)
    for loc, energy, size in [((-5,4,15),2200,9), ((5,-3,12),1000,8)]:
        bpy.ops.object.light_add(type='AREA', location=loc)
        light=bpy.context.object
        light.data.energy=energy
        light.data.shape='DISK'
        light.data.size=size
        light.rotation_euler=(Vector((0,0,0))-light.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.object.camera_add()
    cam=bpy.context.object
    cam.name='One uninterrupted camera move'
    cam.data.type='ORTHO'
    s.camera=cam
    # Bake camera motion so the saved .blend plays without Python callbacks.
    for frame in range(1,s.frame_end+1):
        t=(frame-1)/fps
        p=ease((t-.6)/6.3)
        cam.location=(-1.2*(1-p), -9*(1-p)-1.7*p, 26)
        target=Vector((-1.2*(1-p),0,0))
        cam.rotation_euler=(target-cam.location).to_track_quat('-Z','Y').to_euler()
        cam.data.ortho_scale=15.6+4.2*p
        cam.keyframe_insert(data_path='location',frame=frame)
        cam.keyframe_insert(data_path='rotation_euler',frame=frame)
        cam.data.keyframe_insert(data_path='ortho_scale',frame=frame)
    section=profile()
    green_path=resample(False,301)
    green_stroke=ribbon('Green continuous ribbon',green_path,green,section,fps,end=3.9)
    gold_path=resample(True,301)
    gold_segments=[]
    # Equal arc-length dashes preserve the reference's visual rhythm.
    for index, begin in enumerate(range(0,297,15)):
        chunk=gold_path[begin:begin+9]
        start=-.75+3.75*begin/300
        dash=ribbon(f'Gold dashed ribbon {index:02d}',chunk,gold,section,fps,
                    start=start,end=start+.24)
        # The tip and all dashes share one reveal controller.
        dash.data.animation_data_clear()
        gold_segments.append((dash,begin,len(chunk)-1))
    # Match the arrows at the start of the source animation. Their tips leave
    # the frame naturally once the two curves finish drawing.
    heads=[(arrowhead('Green leading arrow',green),green_path,-2.05),
           (arrowhead('Gold leading arrow',gold),gold_path,-2.9)]
    for frame in range(1,s.frame_end+1):
        s.frame_set(frame)
        amount=green_stroke.data.bevel_factor_end
        bend=green_stroke.data.shape_keys.key_blocks['Reference curve'].value
        for dash,begin,length in gold_segments:
            dash.data.bevel_factor_end=min(1,max(0,(amount*300-begin)/length))
            dash.data.keyframe_insert(data_path='bevel_factor_end',frame=frame)
        for head,pts,baseline in heads:
            j=min(len(pts)-2,int(amount*(len(pts)-1)))
            fraction=amount*(len(pts)-1)-j
            p=pts[j]*(1-fraction)+pts[j+1]*fraction
            p[1]=baseline+(p[1]-baseline)*bend
            tangent=pts[j+1]-pts[j]
            head.location=p
            head.rotation_euler.z=math.atan2(tangent[1]*bend,tangent[0])
            head.keyframe_insert(data_path='location',frame=frame)
            head.keyframe_insert(data_path='rotation_euler',frame=frame)
    corner(green_path,green,fps)
    title('Series',cfg['series'],(2.65,1.0,.03),.32,accent,
          ROOT/'assets/NotoSans-Regular.ttf',fps,6.25)
    title('Title',cfg['title'],(2.65,-.05,.03),.85,ink,
          ROOT/'assets/NotoSans-Bold.ttf',fps,6.45)
    if cfg.get('guest'):
        title('Guest',cfg['guest'],(2.65,-1.0,.03),.48,ink,
              ROOT/'assets/NotoSans-Regular.ttf',fps,6.8)
    for sec,label in [(0,'Two strokes'),(.8,'The split'),(4.25,'Original green corner'),(6.25,'Title')]:
        s.timeline_markers.new(label,frame=round(sec*fps)+1)
    s['Design']='Reference composition translated to 3D. No axes, labels, chapters, diagrams or explanatory copy.'
    s.frame_set(1)
    bpy.ops.file.pack_all()
    if hasattr(s.render.image_settings,'media_type'):
        s.render.image_settings.media_type='VIDEO'
    s.render.image_settings.file_format='FFMPEG'
    s.render.ffmpeg.format='MPEG4'
    s.render.ffmpeg.codec='H264'
    s.render.ffmpeg.constant_rate_factor='HIGH'
    s.render.filepath='//CVAIL-intro-3D-native.mp4'
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output/'CVAIL-intro-3D.blend'))
    if hasattr(s.render.image_settings,'media_type'):
        s.render.image_settings.media_type='IMAGE'
    s.render.image_settings.file_format='PNG'
    print('INTRO_BUILT',w,h,s.frame_end,flush=True)
    return s


def render(a,s,cfg):
    frames=a.output/'frames'
    frames.mkdir(exist_ok=True)
    selected=[round(t*cfg['fps'])+1 for t in [.7,3,5.5,7.5,min(10.5,cfg['seconds']-.5)]] if a.stills else range(1,s.frame_end+1)
    cache=None
    began=time.monotonic()
    for frame in selected:
        dst=frames/f'{frame:05d}.png'
        if dst.exists():
            continue
        # Camera and all design elements are stationary after 7.5 seconds.
        if frame>round(7.5*cfg['fps'])+1 and cache and not a.stills:
            shutil.copyfile(cache,dst)
            continue
        s.frame_set(frame)
        s.render.filepath=str(dst)
        bpy.ops.render.render(write_still=True)
        if frame>=round(7.5*cfg['fps'])+1:
            cache=dst
        print('INTRO_FRAME',frame,s.frame_end,round(time.monotonic()-began,1),flush=True)
    if a.stills:
        return
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('Install FFmpeg to encode the finished PNG sequence.')
    available=subprocess.run([ffmpeg,'-hide_banner','-encoders'],capture_output=True,text=True,check=True).stdout
    if 'libx264' in available:
        codec=['-c:v','libx264','-crf','18','-preset','medium']
    elif 'libopenh264' in available:
        codec=['-c:v','libopenh264','-b:v','12M']
    else:
        raise RuntimeError('FFmpeg needs libx264 or libopenh264.')
    subprocess.run([ffmpeg,'-y','-hide_banner','-loglevel','error','-framerate',str(cfg['fps']),
                    '-i',str(frames/'%05d.png'),*codec,'-threads','8','-pix_fmt','yuv420p',
                    '-movflags','+faststart','-an',str(a.output/'CVAIL-intro-3D.mp4')],check=True)
    print('INTRO_MP4',a.output/'CVAIL-intro-3D.mp4',flush=True)


if __name__=='__main__':
    a=parse()
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(a.config.read_text())
    if cfg['seconds']<8:
        raise ValueError('Use seconds >= 8 for this timing.')
    scene=build(a,cfg)
    if not a.build_only:
        render(a,scene,cfg)
