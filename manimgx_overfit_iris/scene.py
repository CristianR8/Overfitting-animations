"""CVAIL: a curve memorizes noisy points, misses new ones, and folds into an iris.

All imagery is vector/triangle geometry rendered by ManimGX. No stock video,
generated imagery, labels, axes, or external compositing are used.
"""
from pathlib import Path
import json

import numpy as np
import manimgx as m

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
m.config.background_color = CFG['background']


def ease(x):
    x = np.clip(x, 0., 1.)
    return x*x*x*(10+x*(-15+6*x))


def rgb(color):
    return np.array([int(color[i:i+2], 16)/255 for i in (1, 3, 5)])


def trend(x):
    return .72*np.sin(.63*x)+.07*x


X = np.linspace(-5., 5., 15)
NOISE = np.array([.15, -.50, .72, -.30, .90, -.75, .38,
                  -1., .80, -.42, .63, -.80, .55, -.48, .20])
Y = trend(X)+NOISE
TRAIN = np.column_stack([X, Y, np.zeros(len(X))])
VALID_X = X[[2, 5, 7, 8, 10, 11]]+.18
VALID_Y = trend(VALID_X)


def fitted_y(x, second):
    """Regularized Gaussian basis fit: narrowing kernels memorize sample noise."""
    p = float(ease((second-CFG['fit_start'])/CFG['fit_duration']))
    sigma = 2.3*(1-p)+.34*p
    regularization = .45*(1-p)**3+1e-6
    kernel = np.exp(-.5*((X[:, None]-X[None, :])/sigma)**2)
    weights = np.linalg.solve(kernel+regularization*np.eye(len(X)), Y-.10*X)
    basis = np.exp(-.5*((np.asarray(x)[:, None]-X[None, :])/sigma)**2)
    learned = float(ease((second-2.2)/1.3))
    return .10*np.asarray(x)+learned*(basis@weights)


def arrival(index, second):
    u = float(ease((second-.12-.035*index)/1.7))
    a = 2*np.pi*index/len(X)+.5
    start = np.array([5.1*np.cos(a), 2.8*np.sin(a), 2.3*np.sin(a+.7)])
    c1 = start*np.array([.74, .70, 1.0])+np.array([0., 0., 1.1])
    end = TRAIN[index]
    c2 = end+np.array([-.8*np.cos(a), .6*np.sin(a), .65*np.cos(a)])
    return (1-u)**3*start+3*(1-u)**2*u*c1+3*(1-u)*u*u*c2+u**3*end


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
        self.camera.background_color = CFG['background']
        self.camera.bloom = False
        self.camera.ambient_occlusion = False

        curve = Tube(501, CFG['green'], .072, ring=12)
        beads = [Bead(CFG['gold']) for _ in X]
        trails = [Tube(25, CFG['gold'], .018, ring=6, gloss=False) for _ in X]
        rings = [Ring(CFG['gold'], .022) for _ in VALID_X]
        cores = [Disc(CFG['background'], count=32) for _ in VALID_X]
        errors = [Tube(9, CFG['gold'], .008, ring=6, gloss=False) for _ in VALID_X]
        halos = [Ring(CFG['gold'], .012, count=40) for _ in X]
        geometry = m.Group(*[trail.mesh for trail in trails], curve.mesh,
                           *[bead.mesh for bead in beads], *[halo.mesh for halo in halos],
                           *[line.mesh for line in errors], *[core.mesh for core in cores],
                           *[ring.mesh for ring in rings])
        self.add(geometry)

        wipe = Disc(CFG['green'], count=192)
        self.add_fixed_in_frame_mobjects(wipe.mesh)
        series = m.Text(CFG['series'], font='Noto Sans', font_size=52, color=CFG['gold'])
        series.scale_to_fit_width(10.)
        title = m.Text(CFG['title'], font='Noto Sans', font_size=44,
                       weight='BOLD', color=CFG['background'])
        title.scale_to_fit_width(7.7)
        series_pose, title_pose = FlatPose(series), FlatPose(title)
        self.add_fixed_in_frame_mobjects(series, title)
        greek = []
        for char in CFG['greek_letters']:
            glyph = m.Text(char, font='Noto Sans', font_size=78, color=CFG['gold'])
            glyph.scale_to_fit_height(1.06)
            self.add_fixed_in_frame_mobjects(glyph)
            greek.append((glyph, FlatPose(glyph)))
        logos = []
        for filename, height, center in [
            ('uis-reference.png', .66, [-.80, -3.10, 0]),
            ('cvail-logo.png', .94, [1.02, -3.10, 0]),
        ]:
            image = m.ImageMobject(ROOT/'assets'/filename)
            image.scale_to_fit_height(height)
            self.add_fixed_in_frame_mobjects(image)
            logos.append((image, FlatPose(image), np.array(center)))

        s = np.linspace(0, 1, 501)
        x = -5+10*s
        circle_angle = 2*np.pi*np.arange(len(X))/len(X)-np.pi/2

        def state(second):
            complexity = float(ease((second-CFG['fit_start'])/CFG['fit_duration']))
            collapse = float(ease((second-CFG['collapse_start'])/CFG['collapse_duration']))
            relax = float(ease((second-10.4)/.5))
            hero = 1-float(ease((second-11.35)/.55))
            # The initial oblique orbit settles into a frontal view for the reveal.
            front = float(ease((second-5.4)/2.1))
            dolly = float(ease(second/2.6))
            phi = np.deg2rad(62-17*dolly)*(1-front)
            orbit = float(ease(second/5.4))
            theta = -np.pi/2+np.deg2rad(-32+57*orbit)*(1-front)
            self.camera.phi_tracker.set_value(phi)
            self.camera.theta_tracker.set_value(theta)
            self.camera.gamma_tracker.set_value(np.deg2rad(-4)*(1-front))
            self.camera.focal_distance_tracker.set_value(24.)
            self.camera.zoom_tracker.set_value((1.22-.29*dolly)*(1-front)+front)
            self.camera.frame.move_to([0, 0, 0])

            y = fitted_y(x, second)
            graph = np.column_stack([x, y, np.zeros(len(x))])
            closed = float(ease((second-10.2)/.6))
            angle = 2*np.pi*(14/15+closed/15)*s-np.pi/2
            radius = 2.25+.15*(1-relax)*np.cos(15*(angle+np.pi/2))
            iris = np.column_stack([radius*np.cos(angle), radius*np.sin(angle),
                                    .10*(1-relax)*np.sin(2*angle)])
            points = graph*(1-collapse)+iris*collapse
            draw = float(ease((second-1.65)/1.45))
            # Collapse unrevealed points to the advancing tip, so there is no ghost line.
            if draw < 1:
                sample = np.minimum(s, draw)
                points = np.column_stack([np.interp(sample, s, points[:, i]) for i in range(3)])
            curve.set(points, opacity=hero*float(ease((second-1.65)/.35)), reveal=draw)

            for i, (bead, trail, halo) in enumerate(zip(beads, trails, halos)):
                visible = float(ease((second-.12-.035*i)/.28))
                r = 2.4-.15*relax
                target = np.array([r*np.cos(circle_angle[i]), r*np.sin(circle_angle[i]), .13])
                center = arrival(i, second)*(1-collapse)+target*collapse
                bead.set(center, opacity=visible*hero, scale=.78+.22*visible)
                trail_opacity = .42*visible*(1-float(ease((second-1.5-.035*i)/.65)))
                if trail_opacity > .0001:
                    history = np.linspace(max(0, second-.48), second, 25)
                    trail.set(np.array([arrival(i, t) for t in history]),
                              opacity=trail_opacity, taper=True)
                else:
                    trail.mesh.set_opacity(0)
                pulse = float(ease((second-6.65-.018*i)/.65))
                halo.set(center+np.array([0, 0, -.04]), .15+.22*pulse,
                         opacity=.30*np.sin(np.pi*pulse)*hero*(1-collapse))

            predictions = fitted_y(VALID_X, second)
            for i, (ring, core, line) in enumerate(zip(rings, cores, errors)):
                arrive = float(ease((second-7.05-.11*i)/.45))
                leave = float(ease((second-8.7)/.7))
                opacity = arrive*(1-leave)*hero
                center = np.array([VALID_X[i], VALID_Y[i], .17])
                core.set(center+np.array([0, 0, -.025]), .087, opacity)
                ring.set(center, .105*(.78+.22*arrive), opacity)
                a = np.array([VALID_X[i], predictions[i], .04])
                b = a+(center-a)*arrive
                line.set(np.linspace(a, b, 9), opacity=.32*opacity)

            fill = float(ease((second-CFG['wipe_start'])/CFG['wipe_duration']))
            wipe.set(np.array([0., 0., 0.]), max(1e-5, 12*fill), float(fill > 0))
            lead = CFG['title_start']
            series_in = float(ease((second-lead)/.75))
            title_in = float(ease((second-lead-.18)/.75))
            series.set_opacity(series_in)
            title.set_opacity(title_in)
            series_pose.set(np.array([0., 1.30-.22*(1-series_in), 0.]), .98+.02*series_in)
            title_pose.set(np.array([0., .05-.18*(1-title_in), 0.]), .98+.02*title_in)
            for i, (glyph, pose) in enumerate(greek):
                elapsed = second-CFG['greek_start']-CFG['greek_interval']*i
                enter = float(ease(elapsed/.18))
                leave = 0 if i == len(greek)-1 else float(ease((elapsed-CFG['greek_interval']+.10)/.10))
                glyph.set_opacity(enter*(1-leave))
                pose.set(np.array([0., -1.32-.09*(1-enter), 0.]), .95+.05*enter)
            for i, (image, pose, center) in enumerate(logos):
                p = float(ease((second-12.65-.10*i)/.65))
                image.set_opacity(p)
                pose.set(center+np.array([0, -.12*(1-p), 0]), .97+.03*p)

        state(0)
        self.play(m.UpdateFromAlphaFunc(geometry, lambda _, a: state(a*CFG['duration'])),
                  run_time=CFG['duration'], rate_func=m.linear)
