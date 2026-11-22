#!/usr/bin/env python3
# SPDX-FileCopyrightText: IRedDragonICY
# SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
"""Writes Kelir's golden files: kelir_params.json, the kind table every encoder and decoder must
match, and <name>.bin sections with <name>.json, the plan each must compile to.

framefx's host tests and Pentas's unit tests both hold their code to these files; Pentas keeps a
copy pinned by SHA-256. Change them only together with a minor or major version of the section.
    tests/golden/kelir/make.py
"""

import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
Q_MAX = 32767


def c(name, lo, default, hi, needs=-1):
    return {"name": name, "choices": 0, "lo": lo, "default": default, "hi": hi, "needs": needs}


def d(name, choices):
    return {"name": name, "choices": choices, "lo": 0, "default": 0, "hi": 0, "needs": -1}


# Ordinals are the wire format: append only.
KINDS = [
    ("Brightness/Contrast", "colour", False,
     [c("Exposure", -2, 0, 2), c("Contrast", -1, 0, 1), c("Highlights", -1, 0, 1),
      c("Shadows", -1, 0, 1), c("Gamma", 0.5, 1, 2)]),
    ("Color", "colour", False,
     [c("Temperature", -1, 0, 1), c("Tint", -1, 0, 1), c("Hue", -180, 0, 180),
      c("Saturation", -1, 0, 1), c("Vibrance", -1, 0, 1), c("Tint colour", 0, 0, 360, 6),
      c("Tint intensity", 0, 0, 1)]),
    ("Levels", "colour", False,
     [c("Black", 0, 0, 0.5), c("White", 0.5, 1, 1), c("Midtones", 0.5, 1, 2),
      c("Brightness", -0.5, 0, 0.5)]),
    ("Black & White", "colour", False, []),
    ("Sepia", "colour", False, []),
    ("Night Mode", "colour", False, []),
    ("Colorblind", "colour", False, [d("Type", 3)]),
    ("Sharpen", "detail", False, [c("Ignore film grain", 0, 0, 1), d("Quality", 2)]),
    ("Details", "detail", False,
     [c("Clarity", -1, 0, 1), c("HDR toning", 0, 0, 1), c("Bloom", 0, 0, 1),
      c("Bloom threshold", 0.2, 0.75, 1, 2)]),
    ("Tilt-Shift", "detail", False,
     [d("Shape", 2), c("Position", 0, 0.5, 1), c("Size", 0.05, 0.3, 1), c("Softness", 0, 0.5, 1),
      c("Angle", -90, 0, 90), c("Blur", 0, 0.5, 1)]),
    ("Painterly", "detail", True, [c("Brush", 0, 0.5, 1), c("Detail", 0, 0.5, 1)]),
    ("Watercolor", "detail", True,
     [c("Blurriness", 0, 0.5, 1), c("Fuzziness", 0, 0.5, 1), c("Pencil detail", 0, 0.5, 1),
      c("Pencil intensity", 0, 0.5, 1)]),
    ("Vignette", "screen", False, [c("Size", 0, 0.5, 1), c("Roundness", 0, 0.5, 1)]),
    ("Film Grain", "screen", False, [c("Size", 0.5, 1, 3), c("Colour", 0, 0, 1)]),
    ("Letterbox", "screen", False, [c("Aspect", 1.33, 2.39, 3)]),
    ("Old Film", "screen", False,
     [c("Dust", 0, 0.5, 1), c("Scratches", 0, 0.5, 1), c("Flicker", 0, 0.3, 1)]),
    ("Retro", "screen", False,
     [c("Pixel size", 1, 4, 16), c("Colour levels", 2, 8, 32), c("Scanlines", 0, 0.5, 1)]),
    ("Halftone", "screen", False, [c("Dot size", 2, 6, 16), d("Ink", 2)]),
]
KIND = {name: i for i, (name, _, _, _) in enumerate(KINDS)}


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def nearest(x):
    """Ties away from zero, as lround and Kotlin's roundToInt round the positive values here."""
    return int(math.floor(x + 0.5))


def encode(p, value):
    if p["choices"]:
        return int(value)
    lo, de, hi = f32(p["lo"]), f32(p["default"]), f32(p["hi"])
    value = f32(value)
    if value >= de:
        return 0 if hi == de else nearest(f32(f32(min(f32((value - de) / (hi - de)), 1.0)) * Q_MAX))
    return 0 if lo == de else -nearest(f32(f32(min(f32((de - value) / (de - lo)), 1.0)) * Q_MAX))


def decode(p, q):
    if p["choices"]:
        return float(q)
    lo, de, hi = f32(p["lo"]), f32(p["default"]), f32(p["hi"])
    if q == Q_MAX:
        return hi
    if q == -Q_MAX:
        return lo
    t = f32(q / Q_MAX)
    v = f32(de + f32(t * f32(hi - de))) if q >= 0 else f32(de + f32(t * f32(de - lo)))
    return min(max(v, lo), hi)


def node(kind, intensity=1.0, enabled=True, **values):
    name, _, _, params = KINDS[KIND[kind]]
    q = [0] * 8
    for i, p in enumerate(params):
        key = p["name"].lower().replace(" ", "_").replace("/", "_")
        if key in values:
            q[i] = encode(p, values.pop(key))
    assert not values, (kind, values)
    return (KIND[kind], enabled, nearest(intensity * 65535), q)


def section(nodes, peek=False, split=False, inspect=False, split_x=0.0, revision=1, minor=0):
    flags = (1 if peek else 0) | (2 if split else 0) | (4 if inspect else 0)
    out = struct.pack("<BBBBHHII", 1, minor, flags, len(nodes), nearest(split_x * 65535), 0,
                      revision, 0)
    for kind, enabled, intensity, q in nodes:
        out += struct.pack("<BBH8h", kind, 1 if enabled else 0, intensity, *q)
    return out


def band(kind):
    return KINDS[kind][1]


def inert(kind, enabled, intensity, values):
    if not enabled or intensity == 0:
        return True
    name = KINDS[kind][0]
    params = KINDS[kind][3]
    at_default = all(values[i] == f32(p["default"]) for i, p in enumerate(params)
                     if p["name"] != "Tint colour")
    if name in ("Brightness/Contrast", "Color", "Levels"):
        return at_default
    if name == "Details":
        return values[0] == 0 and values[1] == 0 and values[2] == 0
    if name == "Tilt-Shift":
        return values[5] == 0
    return False


def plan(nodes, revision, peek, split, inspect, split_x):
    bands = {"detail": [], "colour": [], "screen": []}
    for kind, enabled, intensity, q in nodes:
        params = KINDS[kind][3]
        values = [decode(p, q[i]) for i, p in enumerate(params)]
        if inert(kind, enabled, intensity, values):
            continue
        bands[band(kind)].append({
            "kind": KINDS[kind][0],
            "intensity": round(intensity / 65535, 6),
            "values": [round(v, 6) for v in values],
        })
    return {"revision": revision, "peek": peek, "split": split, "inspect": inspect,
            "split_x": round(nearest(split_x * 65535) / 65535, 6), **bands}


VECTORS = {
    "empty": dict(nodes=[]),
    "flags": dict(nodes=[], peek=True, split=True, inspect=True, split_x=0.5, revision=0xdeadbeef),
    "colour_band": dict(nodes=[
        node("Brightness/Contrast", exposure=0.5, contrast=0.2, shadows=0.3, gamma=1.2),
        node("Color", 0.8, temperature=0.3, vibrance=0.5, tint_colour=200.0, tint_intensity=0.25),
        node("Levels", black=0.05, white=0.95, midtones=1.1),
        node("Black & White", 0.4),
        node("Sepia", 0.3),
        node("Night Mode", 0.5),
        node("Colorblind", type=1),
    ], revision=7),
    "detail_band": dict(nodes=[
        node("Sharpen", 0.7, ignore_film_grain=0.5, quality=1),
        node("Color", saturation=0.25),
        node("Details", clarity=0.4, bloom=0.3, bloom_threshold=0.6),
        node("Painterly", brush=0.8, detail=0.2),
    ], revision=8),
    "screen_band": dict(nodes=[
        node("Vignette", 0.6, size=0.7, roundness=1.0),
        node("Film Grain", 0.5, size=2.0, colour=0.3),
        node("Brightness/Contrast", contrast=-0.25),
        node("Letterbox", aspect=2.39),
        node("Retro", pixel_size=6.0, colour_levels=4.0, scanlines=0.8),
    ], revision=9),
    "max_stack": dict(nodes=[
        node("Tilt-Shift", shape=1, position=0.4, size=0.25, softness=0.6, angle=-30.0, blur=0.9),
        node("Halftone", 0.75, dot_size=8.0, ink=1),
        node("Sharpen", 0.5),
        node("Old Film", dust=0.9, scratches=0.2, flicker=0.6),
        node("Watercolor", blurriness=0.3, fuzziness=0.7, pencil_detail=0.9, pencil_intensity=0.4),
        node("Vignette"),
        node("Letterbox", 0.5, aspect=1.85),
        node("Brightness/Contrast", exposure=-1.0, highlights=-0.5),
    ], split=True, split_x=0.3, revision=10),
    "inert_nodes": dict(nodes=[
        node("Brightness/Contrast"),
        node("Color", tint_colour=120.0),
        node("Details"),
        node("Tilt-Shift", blur=0.0),
        node("Sepia", 0.0),
        node("Vignette", enabled=False),
    ], revision=11),
}


def main():
    table = [{"name": n, "band": b, "style": s, "params": p} for n, b, s, p in KINDS]
    with open(os.path.join(HERE, "..", "kelir_params.json"), "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    for name, v in VECTORS.items():
        args = dict(peek=False, split=False, inspect=False, split_x=0.0, revision=1)
        args.update(v)
        nodes = args.pop("nodes")
        with open(os.path.join(HERE, name + ".bin"), "wb") as f:
            f.write(section(nodes, **args))
        with open(os.path.join(HERE, name + ".json"), "w") as f:
            json.dump(plan(nodes, args["revision"], args["peek"], args["split"], args["inspect"],
                           args["split_x"]), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
