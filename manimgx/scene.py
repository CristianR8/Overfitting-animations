"""CVAIL / two travelling waves in 3D, separating into the original flat curves."""
from pathlib import Path
import json

import numpy as np
import manimgx as m

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
m.config.background_color = CFG['background']


def ease(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3 - 2 * value)


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def motion_time(second):
    start=CFG['flight_seconds']
    if second<=start:
        return second
    u=float(np.clip((second-start)/CFG['split_seconds'],0,1))
    # Integral of (1 - smoothstep): travel slows to rest without reversing.
    return start+CFG['split_seconds']*(u-u**3+.5*u**4)


def rgb(hex_value):
    return np.array([int(hex_value[i:i+2], 16)/255 for i in (1, 3, 5)])


def opening(second):
    return float(ease((second-CFG['front_hold'])/CFG['opening_seconds']))


def signal_start(second=0.):
    flatten=float(ease((second-CFG['flight_seconds'])/CFG['split_seconds']))
    return CFG['signal_start']*(1-flatten)


def reveal(second):
    """A continuous left-to-right draw, with matching speed at every keyframe."""
    knots=CFG['draw_keyframes']
    if second<=knots[0][0]:
        return knots[0][1]
    for (t0,s0,v0),(t1,s1,v1) in zip(knots,knots[1:]):
        if second<=t1:
            h=t1-t0
            u=(second-t0)/h
            return ((2*u**3-3*u**2+1)*s0+(u**3-2*u**2+u)*h*v0
                    +(-2*u**3+3*u**2)*s1+(u**3-u**2)*h*v1)
    return knots[-1][1]



def camera_angles(second,flatten):
    opened=opening(second)
    orbit=float(ease((second-CFG['orbit_start'])/CFG['orbit_seconds']))
    phi=np.deg2rad(64)*opened*(1-flatten)
    theta=-np.pi/2+2*np.pi*orbit
    return phi,theta



def camera_center(second,flatten):
    return np.array([1.25*motion_time(second)*(1-flatten),
                     -.45*np.sin(np.pi*flatten)**2,0])


def camera_zoom(second,flatten,meshes):
    """Fit the 3D orbit, then lock the frontal camera so the tips can exit right."""
    phi,theta=camera_angles(second,flatten)
    view=np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
    right=np.array([-np.sin(theta),np.cos(theta),0])
    up=np.cross(view,right)
    points=np.vstack([mesh.points for mesh in meshes])-camera_center(second,flatten)
    perspective=20/(20-points@view)
    horizontal=(points@right)*perspective
    vertical=(points@up)*perspective
    fit=min(1.,7.04/max(np.max(np.abs(horizontal)),1e-6),
            3.92/max(np.max(np.abs(vertical)),1e-6))
    nominal=1-.18*opening(second)*(1-flatten)
    locked=float(ease((flatten-.75)/.25))
    return min(nominal,fit)*(1-locked)+locked


def line_colors(samples,second,flatten,gold=False):
    base=rgb(CFG['gold'] if gold else CFG['green'])
    return np.tile(base,(len(samples),1))


def flat_path(s, gold=False):
    """The reference's green rising curve and gold rising-then-falling curve."""
    s = np.asarray(s)
    x = 17.2 * (s - .5)
    q=sigmoid((s-.40)*16)
    y = -1.8 + 5.4*q
    if gold:
        # A true normal offset gives both paths the same curvature and an
        # even distance BEFORE their intended bifurcation.
        slope=5.4*16*q*(1-q)/17.2
        length=np.sqrt(1+slope*slope)
        x=x+.55*slope/length
        y=y-.55/length
        divergence=ease((s-.30)/.40)
        y=y-6.3*divergence
    return np.column_stack([x, y, np.zeros_like(s)])


def trajectory(s, second, flatten, gold=False):
    """A translating sinusoidal flight; its depth continuously disappears."""
    s = np.asarray(s)
    phase_time = motion_time(second)
    # Both signals use exactly the same phase in space and time.
    phase = 2*np.pi*(CFG['wave_cycles']*s - CFG['wave_speed']*phase_time)
    drift = 1.25 * phase_time
    y_wave=np.sin(phase)
    z_wave=np.cos(phase)
    flight = np.column_stack([
        17.2*(s-.5) + drift,
        CFG['wave_amplitude']*y_wave*opening(second) + (-.62 if gold else .62),
        (CFG['depth_amplitude']*z_wave + (.09 if gold else -.09))*opening(second),
    ])
    return flight*(1-flatten) + flat_path(s, gold)*flatten


def topology(n, ring=12):
    faces = []
    for i in range(n-1):
        for j in range(ring):
            a, b = i*ring+j, i*ring+(j+1)%ring
            c, d = (i+1)*ring+j, (i+1)*ring+(j+1)%ring
            faces.extend([(a,c,b), (b,c,d)])
    # End caps, using the first ring vertex as the fan anchor.
    for j in range(1,ring-1):
        faces.append((0,j+1,j))
        a=(n-1)*ring
        faces.append((a,a+j,a+j+1))
    return np.array(faces, dtype=np.int64)


class Cord:
    """A real triangle mesh in space. At the end its z extent is exactly zero."""
    def __init__(self, samples, color, radius):
        self.s = np.asarray(samples)
        self.base = rgb(color)
        self.radius = radius
        self.angle = np.arange(12)*2*np.pi/12
        self.mesh = m.MeshMobject(
            np.zeros((len(self.s)*12,3)), topology(len(self.s)),
            color=color, fill_opacity=1, stroke_width=0,
            shade_in_3d=False,
        )

    def update(self, second, flatten, gold=False):
        samples=self.s
        if gold:
            elapsed=max(0,second-CFG['flight_seconds']-CFG['split_seconds'])
            # Continue travelling along the final 2D path, with a soft start.
            shift=CFG.get('dash_flow_speed',.025)*(elapsed-.35*(1-np.exp(-elapsed/.35)))
            shift-=np.floor(self.s[0]+shift)
            samples=self.s+shift
        visible=np.clip((reveal(second)-samples)*90,0,1)*np.clip((samples-signal_start(second))*90,0,1)
        samples=np.clip(samples,signal_start(second),reveal(second))
        centers = trajectory(samples, second, flatten, gold)
        tangent = np.gradient(centers, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-10)
        normal = np.cross(tangent, np.array([0.,0.,1.]))
        normal /= np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
        binormal = np.cross(tangent,normal)
        radius=self.radius*(1+.85*(1-flatten))
        offset = radius*(normal[:,None,:]*np.cos(self.angle)[None,:,None]
                              + binormal[:,None,:]*np.sin(self.angle)[None,:,None])
        offset[:,:,2] *= 1-flatten
        self.mesh.points = (centers[:,None,:]+offset).reshape(-1,3)
        # A restrained cylindrical highlight fades out with the geometric depth.
        facing=.5+.5*np.cos(self.angle-.8)
        shade = .34+.66*facing
        bases=line_colors(samples,second,flatten,gold)
        colors = bases[:,None,:]*shade[None,:,None] + (1-bases[:,None,:])*.16*facing[None,:,None]**9
        colors = colors*(1-flatten) + bases[:,None,:]*flatten
        rgba = np.column_stack([colors.reshape(-1,3),np.ones(len(self.s)*12)])
        rgba[:,3]*=np.repeat(visible,12)
        self.mesh.paint = self.mesh.paint.but(fill=rgba)


class ArrowHead:
    def __init__(self,color,radius):
        self.base=rgb(color)
        self.radius=radius
        # A triangular prism faces the camera and points along the projected
        # forward tangent. It stays readable when the tube travels in depth.
        faces=[(0,1,2),(5,4,3),(0,3,4),(0,4,1),(1,4,5),(1,5,2),(2,5,3),(2,3,0)]
        self.mesh=m.MeshMobject(np.zeros((6,3)),np.array(faces),
                                color=color,fill_opacity=1,stroke_width=0,
                                shade_in_3d=False)

    def update(self,second,flatten,gold=False):
        end=reveal(second)
        pair=trajectory(np.array([end-.002,end]),second,flatten,gold)
        tangent=pair[1]-pair[0]
        tangent/=np.linalg.norm(tangent)
        phi,theta=camera_angles(second,flatten)
        view=np.array([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)])
        eye=camera_center(second,flatten)+20*view
        ray=eye-pair[1]
        # Removing a component along this local viewing ray preserves the
        # projected tangent under perspective, unlike an orthogonal projection.
        forward=tangent-np.dot(tangent,view)/np.dot(ray,view)*ray
        if np.linalg.norm(forward)<1e-5:
            forward=np.array([1.,0.,0.])-view[0]*view
        forward/=np.linalg.norm(forward)
        side=np.cross(view,forward)
        side/=np.linalg.norm(side)
        triangle=np.array([pair[1]+.20*forward,
                           pair[1]-.43*forward+self.radius*1.35*side,
                           pair[1]-.43*forward-self.radius*1.35*side])
        depth=.045*(1-flatten)*view
        points=np.vstack([triangle+depth,triangle-depth])
        self.mesh.points=points
        base=line_colors(np.array([end]),second,flatten,gold)[0]
        front=base*(.94+.06*flatten)+(1-base)*.05*(1-flatten)
        back=base*(.48+.52*flatten)
        colors=np.vstack([np.tile(front,(3,1)),np.tile(back,(3,1))])
        opacity=1-float(ease((second-CFG['arrow_hide_start'])/.25))
        self.mesh.paint=self.mesh.paint.but(fill=np.column_stack([colors,np.full(6,opacity)]))


class FlatPose:
    """Absolute poses for logos and glyphs; never accumulate frame-to-frame transforms."""
    def __init__(self,mob):
        self.mob=mob
        center=mob.get_center()
        self.parts=[(part,part.points.copy()-center) for part in mob.get_family() if len(part.points)]

    def set(self,center,scale=1.0,angle=0.0,xscale=1.0):
        c,s=np.cos(angle),np.sin(angle)
        rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
        for part,points in self.parts:
            part.points=((points*np.array([scale*xscale,scale,scale]))@rotation.T)+center


def dash_ranges(count=31, ratio=.54):
    # Uniform arc-length dashes in the final flat composition.
    s = np.linspace(0,1,4000)
    p = flat_path(s,True)
    arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    total = arc[-1]
    ranges=[]
    for i in range(count):
        a=(i+.10)*total/count
        b=(i+.10+ratio)*total/count
        ranges.append(np.linspace(np.interp(a,arc,s),np.interp(b,arc,s),9))
    return ranges


class CVAILIntro(m.ThreeDScene):
    def construct(self):
        self.camera.background_color=CFG['background']
        self.camera.bloom=False
        self.camera.ambient_occlusion=False
        green=Cord(np.linspace(0,1,241),CFG['green'],.063)
        gold=[Cord(samples,CFG['gold'],.052) for samples in dash_ranges()]
        green_arrow=ArrowHead(CFG['green'],.24)
        gold_arrow=ArrowHead(CFG['gold'],.21)
        lower=flat_path(np.linspace(0,1,181))
        n=len(lower)
        vertices=np.vstack([lower.copy(),lower.copy()])
        triangles=np.array([face for i in range(n-1)
                            for face in [(i,i+1,n+i),(i+1,n+i+1,n+i)]])
        corner=m.MeshMobject(vertices,triangles,color=CFG['green'],fill_opacity=0,
                              stroke_width=0,shade_in_3d=False)
        geometry=m.Group(corner,green.mesh,*[cord.mesh for cord in gold],green_arrow.mesh,gold_arrow.mesh)
        self.add(geometry)

        series=m.Text(CFG['series'],font='Noto Sans',font_size=32,color=CFG['gold'])
        series.scale_to_fit_width(4.9)
        series.move_to([2.0,.95,0])
        title=m.Text(CFG['title'],font='Noto Sans',font_size=46,
                     weight='BOLD',color=CFG['green'])
        title.scale_to_fit_width(5.4)
        title.move_to([2.0,0,0])
        title.set_opacity(0)
        series.set_opacity(0)
        self.add_fixed_in_frame_mobjects(series,title)
        title_pose=FlatPose(title)
        series_pose=FlatPose(series)
        logos=[]
        for filename,height,center,start in [
            ('uis-reference.png',.76,[-5.88,3.15,0],9.0),
            ('cvail-logo.png',.99,[-4.19,3.15,0],9.18),
        ]:
            image=m.ImageMobject(ROOT/'assets'/filename)
            image.scale_to_fit_height(height)
            pose=FlatPose(image)
            image.set_opacity(0)
            self.add_fixed_in_frame_mobjects(image)
            logos.append((image,pose,np.array(center),start))
        greek=[]
        for char in CFG['greek_letters']:
            back=m.Text(char,font='Noto Sans',font_size=78,color=CFG['gold'])
            front=m.Text(char,font='Noto Sans',font_size=78,color=CFG['green'])
            back.scale_to_fit_height(.87)
            front.scale_to_fit_height(.87)
            back.set_opacity(0)
            front.set_opacity(0)
            self.add_fixed_in_frame_mobjects(back,front)
            greek.append((back,FlatPose(back),front,FlatPose(front)))
        guest=None
        if CFG.get('guest'):
            guest=m.Text(CFG['guest'],font='Noto Sans',font_size=30,color=CFG['green'])
            guest.move_to([2.1,-.75,0]).set_opacity(0)
            self.add_fixed_in_frame_mobjects(guest)

        def state(second):
            start=CFG['flight_seconds']
            flatten=float(ease((second-start)/CFG['split_seconds']))
            green.update(second,flatten)
            for cord in gold:
                cord.update(second,flatten,True)
            green_arrow.update(second,flatten)
            gold_arrow.update(second,flatten,True)

            # The same camera turn and geometry blend end together: truly front/2D.
            phase_time=motion_time(second)
            opened=opening(second)
            phi,theta=camera_angles(second,flatten)
            self.camera.phi_tracker.set_value(phi)
            self.camera.theta_tracker.set_value(theta)
            self.camera.gamma_tracker.set_value(0)
            self.camera.focal_distance_tracker.set_value(20)
            self.camera.zoom_tracker.set_value(camera_zoom(second,flatten,
                [green.mesh,*[cord.mesh for cord in gold],green_arrow.mesh,gold_arrow.mesh]))
            self.camera.frame.move_to(camera_center(second,flatten))

            fill=float(ease((second-8.8)/.6))
            verts=vertices.copy()
            verts[:,2]=-.008
            verts[n:,1]=lower[:,1]+(5.5-lower[:,1])*fill
            corner.points=verts
            corner.set_opacity(fill)
            series_in=float(ease((second-9.0)/.55))
            title_in=float(ease((second-9.15)/.6))
            series.set_opacity(series_in)
            title.set_opacity(title_in)
            series_pose.set(np.array([2.0,.95-.10*(1-series_in),0]))
            title_pose.set(np.array([2.0,-.12*(1-title_in),0]),scale=.96+.04*title_in)
            for image,pose,center,logo_start in logos:
                p=float(ease((second-logo_start)/.65))
                image.set_opacity(p)
                pose.set(center+np.array([-.12*(1-p),0,0]),scale=.88+.12*p,
                         angle=np.deg2rad(-5)*(1-p))
            for i,(back,back_pose,front,front_pose) in enumerate(greek):
                elapsed=second-CFG['greek_start']-i*CFG['greek_interval']
                enter=float(ease(elapsed/.16))
                leave=0 if i==len(greek)-1 else float(ease((elapsed-CFG['greek_interval']+.10)/.10))
                opacity=enter*(1-leave)
                back.set_opacity(opacity)
                front.set_opacity(opacity)
                if opacity>0:
                    last=i==len(greek)-1
                    size=(1.18 if last else 1.0)*(1+.10*np.sin(np.pi*enter))
                    center=np.array([2.0,-1.10+.06*(1-enter),0])
                    angle=np.deg2rad(-9)*(1-enter)
                    front_pose.set(center,scale=size,angle=angle,xscale=.72+.28*enter)
                    back_pose.set(center+np.array([.045,-.035,0]),scale=size,angle=angle,xscale=.72+.28*enter)
            if guest:
                guest.set_opacity(float(ease((second-8.85)/.6)))

        state(0)
        self.play(m.UpdateFromAlphaFunc(geometry,lambda _,alpha:state(alpha*CFG['duration'])),
                  run_time=CFG['duration'],rate_func=m.linear)
