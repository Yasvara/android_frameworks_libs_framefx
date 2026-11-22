#!/usr/bin/env python3
# SPDX-FileCopyrightText: IRedDragonICY
# SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
"""Compiles the Luwes and Kelir shaders to SPIR-V into luwes/shaders/spirv.inc and
kelir/shaders/spirv.inc.

The tree has no shader compiler module, so the SPIR-V is checked in. Each module records the
FNV-1a hash of its sources; libframefx static_asserts it against the sources it embeds, so a
shader edited without running this fails the build. Also compiles GLES variants, which the
driver builds at run time, to catch errors here first.

Needs glslangValidator and spirv-val on PATH. Run from anywhere:
    frameworks/libs/framefx/tools/gen_spirv.py
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADERS = os.path.join(ROOT, "luwes", "shaders")
OUT = os.path.join(SHADERS, "spirv.inc")
KELIR_SHADERS = os.path.join(ROOT, "kelir", "shaders")
KELIR_OUT = os.path.join(KELIR_SHADERS, "spirv.inc")

VULKAN_PRELUDE = """#version 450
#define BINDING(vk, gl) set = 0, binding = vk
#define VERTEX_ID gl_VertexIndex
#define PARAMS layout(push_constant) uniform Params { vec4 p[4]; } params
#define PARAMS8 layout(push_constant) uniform Params { vec4 p[8]; } params
#define P(i) params.p[i]
"""

# Must match kGlesPrelude in luwes/shaders.cpp.
GLES_PRELUDE = """#version 320 es
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp image2D;
#define BINDING(vk, gl) binding = gl
#define VERTEX_ID gl_VertexID
#define PARAMS uniform vec4 uParams[4]
#define PARAMS8 uniform vec4 uParams[8]
#define P(i) uParams[i]
"""

# (symbol, source, defines, SPIR-V target); None skips Vulkan.
VULKAN = [
    ("kFullscreenVert", "fullscreen.vert", "", "spirv1.0"),
    ("kPyramidFrag", "pyramid.frag", "", "spirv1.0"),
    ("kMotionComp", "motion.comp", "#define SUBGROUPS\n", "spirv1.3"),
    ("kOutputFrag", "output.frag", "", "spirv1.0"),
    ("kUpscaleFrag", "upscale.frag", "", "spirv1.0"),
    ("kHudFrag", "hud.frag", "", "spirv1.0"),
    ("kDepthFrag", "depth.frag", "", "spirv1.0"),
    ("kAoFrag", "ao.frag", "", "spirv1.0"),
    ("kFocusFrag", "focus.frag", "", "spirv1.0"),
    ("kSceneFrag", "scene.frag", "", "spirv1.0"),
    ("kSunFrag", "sun.frag", "", "spirv1.0"),
    ("kRaysComp", "rays.comp", "", "vulkan1.2"),
    ("kPositionsComp", "positions.comp", "", "spirv1.0"),
    ("kHistoryComp", "history.comp", "", "spirv1.0"),
    ("kTemporalFrag", "temporal.frag", "", "spirv1.0"),
    ("kTemporalUpFrag", "temporal_up.frag", "", "spirv1.0"),
    ("kGiFrag", "gi.frag", "", "spirv1.0"),
    ("kTransposeFrag", "transpose.frag", "", "spirv1.0"),
    ("kYcbcrFrag", "ycbcr.frag", "", "spirv1.0"),
    ("kHintFrag", "hint.frag", "", "spirv1.0"),
    ("kDenseFrag", "dense.frag", "", "spirv1.0"),
    ("kSplatComp", "splat.comp", "", "spirv1.0"),
]

GLES = ["fullscreen.vert", "pyramid.frag", "motion.comp", "output.frag", "upscale.frag", "hud.frag"]

# Kelir: Luwes's output pass and Kelir's own share a push block, Kelir's frame constants after
# the first 64 bytes.
KELIR_VULKAN_PRELUDE = """#version 450
#define BINDING(vk, gl) set = 0, binding = vk
#define KELIR_SAMPLER(set_, binding_) layout(set = set_, binding = binding_) uniform
#define KELIR_PLAN layout(set = 1, binding = 0, std140) uniform KelirPlan { vec4 v[16]; } kelirPlan
#define PLAN(i) kelirPlan.v[i]
layout(push_constant) uniform Push { vec4 p[4]; vec4 k[4]; } push;
#define PARAMS const int kParamsAt = 0
#define P(i) push.p[i]
#define KELIR_FRAME const int kKelirFrameAt = 64
#define FRAME(i) push.k[i]
#define VERTEX_ID gl_VertexIndex
"""

# Must match kGlesPrelude in kelir/shaders.cpp.
KELIR_GLES_PRELUDE = """precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;
#define KELIR_SAMPLER(set_, binding_) uniform
#define KELIR_PLAN uniform vec4 uKelirPlan[16]
#define PLAN(i) uKelirPlan[i]
#define KELIR_FRAME uniform vec4 uKelirFrame[4]
#define FRAME(i) uKelirFrame[i]
#define PARAMS uniform vec4 uParams[4]
#define P(i) uParams[i]
#define VERTEX_ID gl_VertexID
"""

KELIR_SNIPPETS = [
    ("sharpen.glsl", "SHARPEN"), ("details.glsl", "DETAILS"), ("tiltshift.glsl", "TILT_SHIFT"),
    ("painterly.glsl", "PAINTERLY"), ("watercolor.glsl", "WATERCOLOR"),
    ("vignette.glsl", "VIGNETTE"), ("grain.glsl", "FILM_GRAIN"), ("letterbox.glsl", "LETTERBOX"),
    ("oldfilm.glsl", "OLD_FILM"), ("retro.glsl", "RETRO"), ("halftone.glsl", "HALFTONE"),
]

# (symbol, sources in order, defines); kelir/shaders.cpp hashes the same sources in this order.
KELIR_VULKAN = [
    ("kKelirFrag",
     ["spec.glsl", "common.glsl", "resources.glsl"] + [n for n, _ in KELIR_SNIPPETS] +
     ["stage_b.glsl", "finish.glsl", "kelir.frag"],
     "".join(f"#define KELIR_HAS_{d}\n" for _, d in KELIR_SNIPPETS)),
    ("kPyramidFrag", ["common.glsl", "pyramid.frag"], ""),
]

SCREEN = ("VIGNETTE", "FILM_GRAIN", "LETTERBOX", "OLD_FILM", "RETRO", "HALFTONE")

# The spec's constants for a GLES check: every kind in reach of the dispatch.
KELIR_GLES_CONSTANTS = """const int KELIR_READS_LINEAR = {reads};
const int KELIR_WRITES_LINEAR = {writes};
const int KELIR_LUT_ON = 1;
const int KELIR_D0 = 7;
const int KELIR_D1 = 8;
const int KELIR_D2 = 9;
const int KELIR_SHARPEN_PLUS = 1;
const int KELIR_S0 = 12;
const int KELIR_S1 = 13;
const int KELIR_S2 = 16;
const int KELIR_S3 = 17;
const int KELIR_ROLE = {role};
"""


def fnv1a(data):
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def run(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(" ".join(args) + "\n" + r.stdout + r.stderr)


def compile_module(tmp, text, stage, target, out):
    src = os.path.join(tmp, "shader." + stage)
    with open(src, "w") as f:
        f.write(text)
    env, val = {
        "spirv1.0": (["--target-env", "vulkan1.0"], "vulkan1.0"),
        "spirv1.3": (["--target-env", "vulkan1.1"], "vulkan1.1"),
        "spirv1.4": (["--target-env", "vulkan1.1", "--target-env", "spirv1.4"], "vulkan1.1spv1.4"),
        "vulkan1.2": (["--target-env", "vulkan1.2"], "vulkan1.2"),
    }[target]
    # Ray query needs GLSL 4.60.
    if target == "vulkan1.2":
        text = text.replace("#version 450", "#version 460", 1)
        with open(src, "w") as f:
            f.write(text)
    run(["glslangValidator", "-V", *env, "-o", out, src])
    run(["spirv-val", "--target-env", val, out])


def read(directory, name):
    with open(os.path.join(directory, name)) as f:
        return f.read()


def emit(lines, symbol, source_hash, blob):
    words = [int.from_bytes(blob[i:i + 4], "little") for i in range(0, len(blob), 4)]
    lines.append(f"inline constexpr uint32_t {symbol}Source = 0x{source_hash:08x}u;")
    lines.append(f"inline constexpr uint32_t {symbol}[] = {{")
    for i in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{w:08x}u" for w in words[i:i + 8]) + ",")
    lines.append("};")
    lines.append("")


def kelir(tmp):
    lines = [
        "// Generated by tools/gen_spirv.py from kelir/shaders. Do not edit.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace framefx::kelir::spirv {",
        "",
    ]
    snippets = "".join(read(KELIR_SHADERS, n) for n, _ in KELIR_SNIPPETS)
    has = "".join(f"#define KELIR_HAS_{d}\n" for _, d in KELIR_SNIPPETS)
    body = (read(KELIR_SHADERS, "common.glsl") + read(KELIR_SHADERS, "resources.glsl") + snippets +
            read(KELIR_SHADERS, "stage_b.glsl") + read(KELIR_SHADERS, "finish.glsl") +
            read(KELIR_SHADERS, "kelir.frag"))
    for fetch, role in ((False, 0), (True, 0), (False, 1), (True, 2)):
        for reads, writes in ((0, 0), (1, 1), (0, 1)):
            src = os.path.join(tmp, "kelir.frag")
            text = KELIR_GLES_PRELUDE + KELIR_GLES_CONSTANTS.format(reads=reads, writes=writes, role=role) + has
            if fetch:
                # glslang has no framebuffer fetch: the fetch path is checked reading a plain
                # output instead.
                text += "#define KELIR_FETCH\n" + body.replace("inout highp vec4", "out highp vec4")
            else:
                text += body
            with open(src, "w") as g:
                g.write("#version 300 es\n" + text)
            run(["glslangValidator", src])
    # Luwes's output pass with Kelir's finish, as luwes/gles.cpp assembles it.
    screens = [(n, d) for n, d in KELIR_SNIPPETS if d in
               ("VIGNETTE", "FILM_GRAIN", "LETTERBOX", "OLD_FILM", "RETRO", "HALFTONE")]
    finish = (KELIR_GLES_PRELUDE + "#define KELIR_FINISH\n" +
              KELIR_GLES_CONSTANTS.format(reads=0, writes=0, role=2) +
              "".join(f"#define KELIR_HAS_{d}\n" for _, d in screens) +
              read(KELIR_SHADERS, "common.glsl") + "".join(read(KELIR_SHADERS, n) for n, _ in screens) +
              read(KELIR_SHADERS, "stage_b.glsl") + read(KELIR_SHADERS, "finish.glsl"))
    src = os.path.join(tmp, "fused.frag")
    with open(src, "w") as g:
        g.write(GLES_PRELUDE + finish + read(SHADERS, "output.frag"))
    run(["glslangValidator", src])
    for name, stage in (("fullscreen.vert", "vert"), ("pyramid.frag", "frag")):
        src = os.path.join(tmp, "gles." + stage)
        with open(src, "w") as g:
            g.write("#version 300 es\n" + KELIR_GLES_PRELUDE)
            if stage == "frag":
                g.write(read(KELIR_SHADERS, "common.glsl"))
            g.write(read(SHADERS if stage == "vert" else KELIR_SHADERS, name))
        run(["glslangValidator", src])
    for symbol, names, defines in KELIR_VULKAN:
        sources = [read(KELIR_SHADERS, n) for n in names]
        spv = os.path.join(tmp, symbol + ".spv")
        compile_module(tmp, KELIR_VULKAN_PRELUDE + defines + "".join(sources), "frag", "spirv1.0",
                       spv)
        with open(spv, "rb") as f:
            emit(lines, symbol, fnv1a("".join(sources).encode()), f.read())
    # Luwes's output pass with Kelir's finish: its screen snippets, then luwes/shaders/output.frag.
    sources = ([read(KELIR_SHADERS, n) for n in ("spec.glsl", "common.glsl")] +
               [read(KELIR_SHADERS, n) for n, d in KELIR_SNIPPETS if d in SCREEN] +
               [read(KELIR_SHADERS, n) for n in ("stage_b.glsl", "finish.glsl")] +
               [read(SHADERS, "output.frag")])
    defines = "#define KELIR_FINISH\n" + "".join(f"#define KELIR_HAS_{d}\n" for d in SCREEN)
    spv = os.path.join(tmp, "kLuwesFinishFrag.spv")
    compile_module(tmp, KELIR_VULKAN_PRELUDE + defines + "".join(sources), "frag", "spirv1.0", spv)
    with open(spv, "rb") as f:
        emit(lines, "kLuwesFinishFrag", fnv1a("".join(sources).encode()), f.read())
    lines.append("}  // namespace framefx::kelir::spirv")
    with open(KELIR_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    lines = [
        "// Generated by tools/gen_spirv.py from luwes/shaders. Do not edit.",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "namespace framefx::luwes::spirv {",
        "",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        for name in GLES:
            with open(os.path.join(SHADERS, name)) as f:
                stage = name.rsplit(".", 1)[1]
                src = os.path.join(tmp, "gles." + stage)
                with open(src, "w") as g:
                    g.write(GLES_PRELUDE + f.read())
                run(["glslangValidator", src])
        for symbol, name, defines, target in VULKAN:
            path = os.path.join(SHADERS, name)
            with open(path, "rb") as f:
                source = f.read()
            stage = name.rsplit(".", 1)[1]
            spv = os.path.join(tmp, symbol + ".spv")
            compile_module(tmp, VULKAN_PRELUDE + defines + source.decode(), stage, target, spv)
            with open(spv, "rb") as f:
                emit(lines, symbol, fnv1a(source), f.read())
        kelir(tmp)
    lines.append("}  // namespace framefx::luwes::spirv")
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
