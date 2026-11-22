// The Kelir pass's inputs: the rendered frame, its quarter-resolution pyramid and the colour band's
// LUT. Stage B fused into another pass has none of them.

KELIR_SAMPLER(0, 0) highp sampler2D kelirSource;
KELIR_SAMPLER(0, 1) highp sampler2D kelirPyramid;
KELIR_SAMPLER(1, 1) highp sampler3D kelirLut;

// The rendered frame at a pixel, encoded; edges clamp.
vec3 kelirTap(ivec2 p) {
    ivec2 limit = ivec2(kelirBufferSize()) - 1;
    vec3 c = texelFetch(kelirSource, clamp(p, ivec2(0), limit), 0).rgb;
    return KELIR_READS_LINEAR != 0 ? kelirEncode(c) : c;
}

// How much of a depth edge the source pixel sits on, 0 where nothing says.
float kelirEdge(ivec2 p) {
    if (FRAME(3).x < 0.5) return 0.0;
    ivec2 limit = ivec2(kelirBufferSize()) - 1;
    return texelFetch(kelirSource, clamp(p, ivec2(0), limit), 0).a;
}

// The pyramid at a buffer uv and level: 0 is a quarter of the frame, each level half the last.
vec3 kelirBlur(vec2 uv, float level) {
    return textureLod(kelirPyramid, uv, level).rgb;
}

// The whole colour band: one tap of the 33^3 LUT, its grid denser toward black (kelir/lut.h).
vec3 kelirColour(vec3 c) {
    vec3 at = pow(clamp(c, 0.0, 1.0), vec3(2.0 / 3.0)) * (32.0 / 33.0) + 0.5 / 33.0;
    vec3 v = textureLod(kelirLut, at, 0.0).rgb;
    return PLAN(14).w > 0.5 ? kelirEncode(v) : v;
}

// Retro's cell centre: the rendered frame through the colour band. Detail is not repeated there:
// a cell is many pixels wide.
vec3 kelirRetroTap(ivec2 p) {
    vec3 s = kelirTap(p);
    return KELIR_LUT_ON != 0 ? kelirColour(s) : s;
}
