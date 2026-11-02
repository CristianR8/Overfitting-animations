"""CVAIL: a decision boundary overfits two classes; a quiet dissolve reveals the title."""
from pathlib import Path
import json
import numpy as np
import manimgx as m

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT/'config.json').read_text())
m.config.background_color = CFG['background']


def ease(x):
    x=np.clip(x,0.,1.)
    return x*x*x*(10+x*(-15+6*x))


def rgb(color):
    return np.array([int(color[i:i+2],16)/255 for i in (1,3,5)])


ROWS=np.array([-2.6,-2.0,-1.6,-.9,-.3,.2,.8,1.2,1.7,2.15,2.6])
TARGETS=np.array([-.10,1.50,1.40,-.40,-1.70,-.45,1.55,.50,-.55,-.35,0.])
# Slightly staggered classes avoid a regular ladder of paired samples.
_green_y=ROWS+np.array([.03,-.05,.04,-.04,.04,-.05,.04,-.03,.04,-.04,0.])
_gold_y=ROWS+np.array([-.02,.04,-.04,.03,-.03,.05,-.04,.03,-.03,.04,0.])
_final_kernel=np.exp(-.5*((ROWS[:,None]-ROWS[None,:])/.29)**2)
_final_weights=np.linalg.solve(_final_kernel+1e-6*np.eye(len(ROWS)),TARGETS-.12*ROWS)
def anchor_boundary(y):
    return .12*y+np.exp(-.5*((y[:,None]-ROWS[None,:])/.29)**2)@_final_weights
_green_x=anchor_boundary(_green_y)-np.array([.38,.45,.35,.44,.40,.36,.42,.37,.43,.36,.40])
_gold_x=anchor_boundary(_gold_y)+np.array([.44,.35,.46,.37,.45,.40,.36,.44,.35,.42,.38])
PAIRS=np.array([point for i in range(len(ROWS)) for point in [[_green_x[i],_green_y[i],0.],[_gold_x[i],_gold_y[i],0.]]])
EXTRA_GREEN=np.array([[-3.65,-2.20,0],[-2.95,-1.30,0],[-3.45,-.40,0],[-2.80,.50,0],[-3.50,1.40,0],[-2.95,2.25,0]])
EXTRA_GOLD=np.array([[3.20,-2.25,0],[3.70,-1.40,0],[2.90,-.50,0],[3.65,.45,0],[3.05,1.35,0],[3.55,2.25,0]])
POINTS=np.vstack([PAIRS,EXTRA_GREEN,EXTRA_GOLD])
LABELS=np.array([i%2 for i in range(len(PAIRS))]+[0]*6+[1]*6)
CHECK_ROWS=np.array([1,2,4,6,8])
CHECK_Y=ROWS[CHECK_ROWS]+.08
CHECK_SIGN=np.sign(TARGETS[CHECK_ROWS])
CHECK_X=.12*CHECK_Y+.40*CHECK_SIGN
CHECK_LABELS=(CHECK_SIGN>0).astype(int)
SCALE=1.05


def boundary_x(y,second):
    p=float(ease((second-CFG['fit_start'])/CFG['fit_duration']))
    sigma=1.65*(1-p)+.29*p
    ridge=.60*(1-p)**3+1e-6
    kernel=np.exp(-.5*((ROWS[:,None]-ROWS[None,:])/sigma)**2)
    weights=np.linalg.solve(kernel+ridge*np.eye(len(ROWS)),TARGETS-.12*ROWS)
    basis=np.exp(-.5*((np.asarray(y)[:,None]-ROWS[None,:])/sigma)**2)
    learned=float(ease((second-2.15)/1.2))
    return .12*np.asarray(y)+learned*(basis@weights)


def arrival(index,second):
    u=float(ease((second-.12-.019*index)/1.45))
    point=POINTS[index]*SCALE
    a=2.399963*index
    offset=np.array([.50*np.cos(a),.42*np.sin(a),1.45+.20*np.sin(a)])
    return point+offset*(1-u)


def tube_faces(n, ring):
    faces = []
    for i in range(n-1):
        for j in range(ring):
            a, b = i*ring+j, i*ring+(j+1)%ring
            c, d = (i+1)*ring+j, (i+1)*ring+(j+1)%ring
            faces.extend([(a, c, b), (b, c, d)])
    for j in range(1, ring-1):
        faces.extend([(0, j+1, j), ((n-1)*ring, (n-1)*ring+j, (n-1)*ring+j+1)])
    return np.array(faces)


class Tube:
    def __init__(self, count, color, radius, ring=10, gloss=True):
        self.count, self.radius, self.ring = count, radius, ring
        self.angle = 2*np.pi*np.arange(ring)/ring
        self.mesh = m.MeshMobject(np.zeros((count*ring, 3)), tube_faces(count, ring),
                                  color=color, fill_opacity=1, stroke_width=0,
                                  shade_in_3d=False)
        base = rgb(color)
        if gloss:
            light = .5+.5*np.cos(self.angle-.65)
            colors = base[None, :]*(.50+.50*light[:, None])
            colors += (1-base[None, :])*.19*light[:, None]**14
        else:
            colors = np.tile(base, (ring, 1))
        self.rgba = np.column_stack([np.tile(colors, (count, 1)), np.ones(count*ring)])

    def set(self, centers, opacity=1., reveal=1., taper=False):
        centers = np.asarray(centers)
        tangent = np.gradient(centers, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
        normal = np.cross(tangent, [0., 0., 1.])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
        binormal = np.cross(tangent, normal)
        radii = self.radius*np.ones(self.count)
        if taper:
            radii *= .3+.7*np.linspace(0, 1, self.count)
        offset = radii[:, None, None]*(normal[:, None, :]*np.cos(self.angle)[None, :, None]
                                      +binormal[:, None, :]*np.sin(self.angle)[None, :, None])
        self.mesh.points = (centers[:, None, :]+offset).reshape(-1, 3)
        rgba = self.rgba.copy()
        visible = np.clip((reveal-np.linspace(0, 1, self.count))*85, 0, 1)
        if reveal >= 1:
            visible[:] = 1
        if taper:
            visible *= np.linspace(0, 1, self.count)**1.2
        rgba[:, 3] = opacity*np.repeat(visible, self.ring)
        self.mesh.paint = self.mesh.paint.but(fill=rgba)


class Bead:
    def __init__(self, color, radius=.125):
        lat, lon = 10, 20
        theta = np.linspace(0, np.pi, lat)
        phi = np.arange(lon)*2*np.pi/lon
        self.unit = np.array([[np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]
                              for t in theta for p in phi])
        faces = tube_faces(lat, lon)
        self.mesh = m.MeshMobject(self.unit*radius, faces, color=color,
                                  stroke_width=0, shade_in_3d=False)
        self.radius = radius
        light = np.array([-.36, .42, .83]); light /= np.linalg.norm(light)
        half = np.array([-.2, .3, .93]); half /= np.linalg.norm(half)
        diffuse = .53+.47*np.maximum(self.unit@light, 0)
        specular = .65*np.maximum(self.unit@half, 0)**25
        base = rgb(color)
        colors = base[None, :]*diffuse[:, None]+(1-base[None, :])*specular[:, None]
        self.rgba = np.column_stack([colors, np.ones(len(self.unit))])

    def set(self, center, opacity=1., scale=1.):
        self.mesh.points = self.unit*self.radius*scale+center
        rgba = self.rgba.copy(); rgba[:, 3] = opacity
        self.mesh.paint = self.mesh.paint.but(fill=rgba)


class Ring:
    def __init__(self, color, width=.017, count=64):
        self.width, self.count = width, count
        a = 2*np.pi*np.arange(count)/count
        self.unit = np.column_stack([np.cos(a), np.sin(a), np.zeros(count)])
        faces = []
        for i in range(count):
            j = (i+1)%count
            faces.extend([(i, j, count+i), (j, count+j, count+i)])
        self.mesh = m.MeshMobject(np.zeros((2*count, 3)), np.array(faces),
                                  color=color, stroke_width=0, shade_in_3d=False)

    def set(self, center, radius, opacity=1.):
        vertices = np.vstack([self.unit*max(0, radius-self.width/2),
                              self.unit*(radius+self.width/2)])+center
        self.mesh.points = vertices
        self.mesh.set_opacity(opacity)


class Disc:
    def __init__(self, color, count=128):
        angle = np.arange(count)*2*np.pi/count
        self.unit = np.vstack([np.zeros((1, 3)),
                               np.column_stack([np.cos(angle), np.sin(angle), np.zeros(count)])])
        faces = np.array([(0, i+1, (i+1)%count+1) for i in range(count)])
        self.mesh = m.MeshMobject(self.unit, faces, color=color,
                                  stroke_width=0, shade_in_3d=False)

    def set(self, center, radius, opacity=1.):
        self.mesh.points = self.unit*radius+center
        self.mesh.set_opacity(opacity)


class FlatPose:
    def __init__(self, mob):
        center = mob.get_center()
        self.parts = [(part, part.points.copy()-center)
                      for part in mob.get_family() if len(part.points)]

    def set(self, center, scale=1.):
        for part, points in self.parts:
            part.points = points*scale+center


class CVAILOverfitIntro(m.ThreeDScene):
    def construct(self):
        self.camera.background_color=CFG['background']
        self.camera.bloom=False
        self.camera.ambient_occlusion=False

        curve=Tube(601,CFG['green'],.048,ring=12)
        colors=[CFG['green'] if label==0 else CFG['gold'] for label in LABELS]
        beads=[Bead(color,radius=.115) for color in colors]
        check_colors=[CFG['green'] if label==0 else CFG['gold'] for label in CHECK_LABELS]
        checks=[Ring(color,.021,count=56) for color in check_colors]
        cores=[Disc(CFG['background'],count=32) for _ in CHECK_X]
        geometry=m.Group(curve.mesh,*[b.mesh for b in beads],
                         *[c.mesh for c in cores],*[r.mesh for r in checks])
        self.add(geometry)

        # A full-frame, stationary layer crossfades the chart into the end card.
        backdrop=m.MeshMobject(np.array([[-8,-5,0],[8,-5,0],[8,5,0],[-8,5,0]]),
                              np.array([[0,1,2],[0,2,3]]),color=CFG['green'],
                              fill_opacity=0,stroke_width=0,shade_in_3d=False)
        self.add_fixed_in_frame_mobjects(backdrop)
        series=m.Text(CFG['series'],font='Noto Sans',font_size=52,color=CFG['gold'])
        series.scale_to_fit_width(10.)
        series.move_to([0,1.30,0])
        title=m.Text(CFG['title'],font='Noto Sans',font_size=44,
                     weight='BOLD',color=CFG['background'])
        title.scale_to_fit_width(7.7)
        title.move_to([0,.05,0])
        self.add_fixed_in_frame_mobjects(series,title)
        greek=[]
        for char in CFG['greek_letters']:
            glyph=m.Text(char,font='Noto Sans',font_size=78,color=CFG['gold'])
            glyph.scale_to_fit_height(1.06)
            glyph.move_to([0,-1.32,0])
            self.add_fixed_in_frame_mobjects(glyph)
            greek.append(glyph)
        logos=[]
        for filename,height,center in [
            ('uis-reference.png',.66,[-.80,-3.10,0]),
            ('cvail-logo.png',.94,[1.02,-3.10,0]),
        ]:
            image=m.ImageMobject(ROOT/'assets'/filename)
            image.scale_to_fit_height(height)
            image.move_to(center)
            self.add_fixed_in_frame_mobjects(image)
            logos.append(image)

        y=np.linspace(-3.05,3.05,601)
        s=np.linspace(0,1,601)

        def state(second):
            # Only a modest opening tilt; everything is still before the ending.
            front=float(ease(second/2.6))
            self.camera.phi_tracker.set_value(np.deg2rad(28)*(1-front))
            self.camera.theta_tracker.set_value(-np.pi/2)
            self.camera.gamma_tracker.set_value(0)
            self.camera.focal_distance_tracker.set_value(24.)
            self.camera.zoom_tracker.set_value(.95+.05*front)
            self.camera.frame.move_to([0,0,0])

            hero=1-float(ease((second-CFG['graph_fade_start'])/CFG['graph_fade_duration']))
            draw=float(ease((second-1.85)/1.45))
            sampled_y=np.interp(np.minimum(s,draw),s,y)
            points=np.column_stack([boundary_x(sampled_y,second),sampled_y,np.zeros(len(y))])*SCALE
            curve.set(points,opacity=hero*float(ease((second-1.85)/.35)),reveal=draw)
            for i,bead in enumerate(beads):
                appear=float(ease((second-.12-.019*i)/.40))
                bead.set(arrival(i,second),opacity=appear*hero,scale=.90+.10*appear)

            for i,(ring,core) in enumerate(zip(checks,cores)):
                arrive=float(ease((second-7.15-.10*i)/.55))
                center=np.array([CHECK_X[i]*SCALE,CHECK_Y[i]*SCALE,.17])
                core.set(center+np.array([0,0,-.02]),.085,arrive*hero)
                ring.set(center,.10,arrive*hero)

            mix=float(ease((second-CFG['background_start'])/CFG['background_duration']))
            backdrop.set_opacity(mix)
            lead=CFG['title_start']
            series.set_opacity(float(ease((second-lead)/CFG['title_duration'])))
            title.set_opacity(float(ease((second-lead-.12)/CFG['title_duration'])))
            for i,glyph in enumerate(greek):
                elapsed=second-CFG['greek_start']-i*CFG['greek_interval']
                enter=float(ease(elapsed/.18))
                leave=0 if i==len(greek)-1 else float(ease((elapsed-CFG['greek_interval']+.10)/.10))
                glyph.set_opacity(enter*(1-leave))
            for i,image in enumerate(logos):
                image.set_opacity(float(ease((second-11.15-.08*i)/1.0)))

        state(0)
        self.play(m.UpdateFromAlphaFunc(geometry,lambda _,a:state(a*CFG['duration'])),
                  run_time=CFG['duration'],rate_func=m.linear)


