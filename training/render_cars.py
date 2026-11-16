"""
render_cars.py - synthetic training pairs for AutoBG reflections (Blender 4.2, Cycles).

For every car model (.glb) and every sampled camera view it renders:
  input_rgba.png   car only on transparent background   (what the AutoBG pipeline gives us)
  target.png       car on a reflective studio floor      (what we want; no floor shadow unless --shadows)
  floor.png        the same floor without the car        (for extracting the reflection layer)
  meta.json        camera, floor look, car size, licence info

Reflection layer for training = target - floor (outside the car), stored separately by the
dataset-prep step, so shadows and reflections stay exactly aligned with the input.

Run:
  blender -b -P render_cars.py -- --models DIR_OR_LIST.csv --out OUT_DIR [options]
"""
import argparse
import csv
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector

# ----------------------------------------------------------------------------- args

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="folder of .glb files, or keep_list.csv (uid column) + --glb_dir")
ap.add_argument("--glb_dir", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=24, help="camera views per car")
ap.add_argument("--floors", type=int, default=2, help="floor looks per view (sampled from 4)")
ap.add_argument("--res", type=int, default=1024, help="image width (height = 3/4)")
ap.add_argument("--samples", type=int, default=48)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--count", type=int, default=10 ** 9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--review", action="store_true",
                help="review mode: one front-3/4 cut-out per car, no floors (for quick visual checks)")
ap.add_argument("--shadows", action="store_true",
                help="let the car cast its shadow on the floor in target renders (default: reflection only)")
ap.add_argument("--elev", default="3,22", help="camera height range in degrees, e.g. 0,45 for low and high shots")
ap.add_argument("--v2", action="store_true",
                help="v2 mode: per view, the car cut-out plus pure-mirror floor passes (with / without the car's "
                     "reflection) under a bright and a dark studio; make_pairs_v2.py lays the reflection onto "
                     "real backdrops")
args = ap.parse_args(argv)
# Blender resolves relative image paths against the drive root, not the current folder: make them absolute
args.out = os.path.abspath(args.out)
args.models = os.path.abspath(args.models)
if args.glb_dir:
    args.glb_dir = os.path.abspath(args.glb_dir)

# ----------------------------------------------------------------------------- floor looks

FLOORS = {
    # name: (base colour RGB, roughness, specular/IOR level)
    "mirror_dark": ((0.035, 0.035, 0.04), 0.02, 1.0),
    "gloss_grey": ((0.18, 0.18, 0.19), 0.06, 0.9),
    "satin_light": ((0.55, 0.55, 0.54), 0.16, 0.7),
    "polished_concrete": ((0.42, 0.41, 0.39), 0.28, 0.5),
}

# ----------------------------------------------------------------------------- scene setup


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    sc.render.use_persistent_data = True          # reuse geometry between the renders of a car
    sc.render.resolution_x = args.res
    sc.render.resolution_y = int(args.res * 3 / 4)
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    try:
        sc.view_settings.view_transform = "AgX"               # filmic-style highlight roll-off
    except TypeError:
        sc.view_settings.view_transform = "Filmic"
    sc.view_settings.look = "None"
    if args.gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
                for dvc in prefs.devices:
                    dvc.use = dvc.type != "CPU"          # GPU only: the CPU slows OptiX down
                sc.cycles.device = "GPU"
                if not getattr(main, "_announced", False):
                    print("RENDER DEVICE:", backend, [dv.name for dv in prefs.devices if dv.use], flush=True)
                    main._announced = True
                break
            except Exception:
                continue
    # neutral studio world
    world = bpy.data.worlds.new("studio")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.6, 0.6, 0.62, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.25
    return sc


def add_lights():
    def area(name, loc, size, energy, rot):
        bpy.ops.object.light_add(type="AREA", location=loc, rotation=rot)
        lt = bpy.context.object
        lt.name = name
        lt.data.size = size
        lt.data.energy = energy
        lt.visible_glossy = False          # no hard softbox rectangles mirrored in the floor
        lt.visible_camera = False
        return lt
    area("top", (0, 0, 7), 8, 900, (0, 0, 0))
    area("left", (-7, -3, 4), 5, 350, (math.radians(60), 0, math.radians(-60)))
    area("right", (7, -3, 4), 5, 350, (math.radians(60), 0, math.radians(60)))


def add_floor():
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    fl = bpy.context.object
    fl.name = "floor"
    mat = bpy.data.materials.new("floor_mat")
    mat.use_nodes = True
    fl.data.materials.append(mat)
    return fl, mat


def set_floor_look(mat, look):
    col, rough, spec = FLOORS[look]
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*col, 1)
    bsdf.inputs["Roughness"].default_value = rough
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = spec
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = spec


# v2: studio brightness for the mirror passes (bright studio -> reflection of a dark car darkens a
# light floor; dark studio -> reflection brightens a dark floor). Matched to backdrops later.
ENVS = {"light": 0.85, "dark": 0.12}


def set_mirror_floor(mat, rough):
    """Floor that only reflects: white metal, so reflected colours are kept as they are."""
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = rough


def add_env_dome():
    """Big white dome that only the floor's reflections see (not the camera, not the car's lighting),
    so the studio brightness changes what the floor reflects without changing how the car looks."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=60, segments=48, ring_count=24, location=(0, 0, 0))
    dome = bpy.context.object
    dome.name = "env_dome"
    mat = bpy.data.materials.new("dome_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    em = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], out.inputs[0])
    dome.data.materials.append(mat)
    dome.hide_render = True
    return dome, em


def set_env(dome, em, level):
    """level None = dome off (normal renders)."""
    if level is None:
        dome.hide_render = True
        return
    em.inputs["Color"].default_value = (1, 1, 1, 1)
    em.inputs["Strength"].default_value = level
    dome.hide_render = False
    dome.visible_camera = False
    dome.visible_diffuse = False
    dome.visible_shadow = False
    dome.visible_glossy = True


# ----------------------------------------------------------------------------- car import


def import_car(path, target_len):
    if not os.path.exists(path):
        raise RuntimeError("glb file missing")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    # the glTF importer adds a log handler on every import and never removes it, so each car
    # would print every line once more than the last: drop them and keep only errors
    import logging
    for name in list(logging.root.manager.loggerDict):
        if "gltf" in name.lower():
            lg = logging.getLogger(name)
            lg.handlers.clear()
            lg.setLevel(logging.ERROR)
    objs = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no meshes")
    # parent everything under one empty so we can transform the car as a whole
    root = bpy.data.objects.new("car_root", None)
    bpy.context.scene.collection.objects.link(root)
    for o in objs:
        if o.parent is None:
            o.parent = root
    bpy.context.view_layer.update()

    def bbox():
        pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
        mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        return mn, mx

    mn, mx = bbox()
    ext = mx - mn
    # sanity: a car is long and low; height must be the smallest extent
    if ext.z > min(ext.x, ext.y) * 1.05:
        raise RuntimeError(f"not upright (extents {tuple(round(e, 2) for e in ext)})")
    # make the car's length run along +Y
    if ext.x > ext.y:
        root.rotation_euler.z += math.pi / 2
        bpy.context.view_layer.update()
        mn, mx = bbox()
        ext = mx - mn
    s = target_len / max(ext.y, 1e-6)
    root.scale = (s, s, s)
    bpy.context.view_layer.update()
    mn, mx = bbox()
    centre = (mn + mx) / 2
    root.location -= Vector((centre.x, centre.y, mn.z))     # centred, wheels on the floor (z = 0)
    bpy.context.view_layer.update()
    mn, mx = bbox()
    return root, meshes, (mx - mn)


# ----------------------------------------------------------------------------- camera


def add_camera():
    cam_data = bpy.data.cameras.new("cam")
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    target = bpy.data.objects.new("cam_target", None)
    bpy.context.scene.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    return cam, target


def sample_view(rng, ext):
    """Dealer-style viewpoints: mostly 3/4 front and rear, some side / straight-on."""
    kind = rng.choices(["front34", "rear34", "side", "front", "rear"], [0.38, 0.27, 0.15, 0.1, 0.1])[0]
    base = {"front34": 35, "rear34": 145, "side": 90, "front": 0, "rear": 180}[kind]
    az = base + rng.uniform(-14, 14)
    if rng.random() < 0.5:
        az = -az                                   # driver or passenger side
    lo, hi = (float(v) for v in args.elev.split(","))
    elev = rng.uniform(lo, hi)                     # degrees above horizontal
    focal = rng.uniform(28, 85)                    # mm on a 36 mm sensor
    return kind, az, elev, focal


def place_camera(cam, target, ext, az, elev, focal, rng):
    cam.data.lens = focal
    cam.data.sensor_width = 36
    L, W, H = ext.y, ext.x, ext.z
    fill = rng.uniform(0.72, 0.86)                 # car fills 72-86 % of the frame width
    a, e = math.radians(az), math.radians(elev)
    # binary-search the distance at which the car's projected width hits the target fill
    hx, hy = W / 2, L / 2
    # car box plus the space its reflection needs below the floor line
    corners = [(sx * hx, sy * hy, z) for sx in (-1, 1) for sy in (-1, 1) for z in (-0.7 * H, 0.0, H)]
    aspect = 3.0 / 4.0                             # frame height / width
    tanh = 18.0 / focal                            # half-width of a 36 mm sensor / focal
    fwd = (-math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), -math.sin(e))
    right = (math.cos(a), -math.sin(a), 0.0)
    up = (fwd[1] * right[2] - fwd[2] * right[1], fwd[2] * right[0] - fwd[0] * right[2],
          fwd[0] * right[1] - fwd[1] * right[0])
    tz = H * 0.15                                  # aim low: car upper-middle, reflection below

    def width_at(dist):
        cx, cy, cz = dist * math.cos(e) * math.sin(a), dist * math.cos(e) * math.cos(a), dist * math.sin(e) + tz
        xs, ys = [], []
        for (px, py, pz) in corners:
            dx, dy, dz = px - cx, py - cy, pz - cz
            depth = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
            xs.append((dx * right[0] + dy * right[1] + dz * right[2]) / max(depth, 1e-3))
            ys.append((dx * up[0] + dy * up[1] + dz * up[2]) / max(depth, 1e-3))
        wfrac = (max(xs) - min(xs)) / (2 * tanh)
        hfrac = (max(ys) - min(ys)) / (2 * tanh * aspect)
        return max(wfrac, hfrac / 0.95)            # whichever axis is tighter decides

    lo_d, hi_d = max(L, W), 60.0
    for _ in range(40):
        mid = 0.5 * (lo_d + hi_d)
        if width_at(mid) > fill:
            lo_d = mid
        else:
            hi_d = mid
    dist = 0.5 * (lo_d + hi_d)
    cam.location = (dist * math.cos(e) * math.sin(a), dist * math.cos(e) * math.cos(a), dist * math.sin(e) + tz)
    target.location = (0, 0, tz)


# ----------------------------------------------------------------------------- render passes


def render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def set_visible(objs, camera=True, glossy=True, diffuse=True, shadow=True):
    for o in objs:
        o.visible_camera = camera
        o.visible_glossy = glossy
        o.visible_diffuse = diffuse
        o.visible_shadow = shadow
        o.hide_render = not (camera or glossy or diffuse or shadow)


# ----------------------------------------------------------------------------- main


def model_list():
    m = args.models
    if m.lower().endswith(".csv"):
        rows = list(csv.DictReader(open(m, newline="", encoding="utf-8")))
        d = args.glb_dir or os.path.join(os.path.dirname(m), "glb")
        return [(r["uid"], os.path.join(d, r["uid"] + ".glb"), r) for r in rows]
    files = sorted(f for f in os.listdir(m) if f.lower().endswith(".glb"))
    return [(os.path.splitext(f)[0], os.path.join(m, f), {}) for f in files]


def main():
    os.makedirs(args.out, exist_ok=True)
    models = model_list()[args.start:args.start + args.count]
    log = open(os.path.join(args.out, "render_log.txt"), "a")
    for mi, (uid, path, info) in enumerate(models):
        rng = random.Random(f"{args.seed}-{uid}")
        car_dir = os.path.join(args.out, uid)
        if os.path.exists(os.path.join(car_dir, "done")):
            continue
        try:
            sc = reset_scene()
            add_lights()
            floor, fmat = add_floor()
            root, meshes, ext = import_car(path, rng.uniform(3.9, 5.2))
            cam, target = add_camera()
            dome, dome_em = add_env_dome()
        except Exception as ex:
            log.write(f"SKIP {uid}: {ex}\n"); log.flush()
            continue
        os.makedirs(car_dir, exist_ok=True)
        for vi in range(1 if args.review else args.views):
            kind, az, elev, focal = ("front34", 35.0, 12.0, 50.0) if args.review else sample_view(rng, ext)
            place_camera(cam, target, ext, az, elev, focal, rng)
            vdir = os.path.join(car_dir, f"v{vi:02d}")
            os.makedirs(vdir, exist_ok=True)
            # 1) input: car only, transparent background (floor invisible to camera, but
            #    still lets the car be lit the same way)
            sc.render.film_transparent = True
            set_visible([floor], camera=False, glossy=False, diffuse=True, shadow=False)
            set_visible(meshes, True, True, True, True)
            render_to(os.path.join(vdir, "input_rgba.png"))
            sc.render.film_transparent = False
            looks = []
            if args.v2 and not args.review:
                # 2) mirror-floor passes. The car is invisible to the camera but visible in reflections,
                #    so  floorcar - floorempty  = exactly what the car's reflection does to the floor.
                rough = rng.choice([0.0, 0.02, 0.05, 0.08, 0.12])
                set_mirror_floor(fmat, rough)
                prev_vt = sc.view_settings.view_transform
                sc.view_settings.view_transform = "Standard"      # plain sRGB, so differences add up
                sc.render.image_settings.color_depth = "16"
                for env, level in ENVS.items():
                    set_env(dome, dome_em, level)
                    set_visible([floor], True, True, False, False)
                    set_visible(meshes, camera=False, glossy=True, diffuse=False, shadow=False)
                    render_to(os.path.join(vdir, f"floorcar_{env}.png"))
                    set_visible(meshes, False, False, False, False)
                    render_to(os.path.join(vdir, f"floorempty_{env}.png"))
                    set_visible(meshes, True, True, True, True)
                sc.view_settings.view_transform = prev_vt
                sc.render.image_settings.color_depth = "8"
                set_env(dome, dome_em, None)
                looks = [f"mirror_r{rough}"]
            elif not args.review:
                looks = rng.sample(list(FLOORS), k=min(args.floors, len(FLOORS)))
            for look in ([] if args.v2 else looks):
                set_floor_look(fmat, look)
                # 2) target: car on the reflective floor. By default the car casts no shadow,
                #    so target - floor is the pure reflection (car pixels come from the input)
                set_visible([floor], True, True, True, True)
                set_visible(meshes, True, True, True, shadow=args.shadows)
                render_to(os.path.join(vdir, f"target_{look}.png"))
                # 3) floor only: car removed (no reflection, no shadow)
                set_visible(meshes, False, False, False, False)
                render_to(os.path.join(vdir, f"floor_{look}.png"))
                set_visible(meshes, True, True, True, True)
            cam_m = cam.matrix_world
            json.dump({
                "uid": uid, "view_kind": kind, "azimuth_deg": az, "elevation_deg": elev,
                "focal_mm": focal, "sensor_mm": 36, "floor_looks": looks,
                "car_extent_m": [ext.x, ext.y, ext.z],
                "cam_matrix_world": [list(r) for r in cam_m],
                "resolution": [sc.render.resolution_x, sc.render.resolution_y],
                "model_info": {k: info.get(k) for k in ("name", "author", "license", "url") if k in info},
            }, open(os.path.join(vdir, "meta.json"), "w"), indent=1)
        open(os.path.join(car_dir, "done"), "w").write("ok")
        log.write(f"OK {uid} ({mi + 1}/{len(models)})\n"); log.flush()


main()
