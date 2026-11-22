// The Kelir pass: one fullscreen triangle over the presented image. Stage A (detail, then the
// colour band as one LUT tap), then the finish (stage B, divider, dither). Detail kinds read the
// rendered frame, never their own output, so each applies its change on top of the one before.
// With frame generation the pass runs as stage A alone: it writes generation's source, which
// the finish never touches.

#ifdef KELIR_FETCH
layout(location = 0) inout highp vec4 kelirOut;
#else
layout(location = 0) out vec4 kelirOut;
#endif

vec3 kelirDetail(int kind, int slot, vec3 c, vec3 src, vec2 px) {
#ifdef KELIR_HAS_SHARPEN
    if (kind == KELIR_SHARPEN) return kelirSharpen(slot, c, src, px);
#endif
#ifdef KELIR_HAS_DETAILS
    if (kind == KELIR_DETAILS) return kelirDetails(slot, c, src, px);
#endif
#ifdef KELIR_HAS_TILT_SHIFT
    if (kind == KELIR_TILT_SHIFT) return kelirTiltShift(slot, c, src, px);
#endif
#ifdef KELIR_HAS_PAINTERLY
    if (kind == KELIR_PAINTERLY) return kelirPainterly(slot, c, src, px);
#endif
#ifdef KELIR_HAS_WATERCOLOR
    if (kind == KELIR_WATERCOLOR) return kelirWatercolor(slot, c, src, px);
#endif
    return c;
}

void main() {
    vec2 px = gl_FragCoord.xy;
#ifdef KELIR_FETCH
    vec3 raw = kelirOut.rgb;
    vec3 src = KELIR_READS_LINEAR != 0 ? kelirEncode(raw) : raw;
#else
    vec3 src = kelirTap(ivec2(px));
#endif
    vec3 c = src;
    if (kelirDisplayUv(px).x >= FRAME(1).z) {
        c = kelirDetail(KELIR_D0, 0, c, src, px);
        c = kelirDetail(KELIR_D1, 1, c, src, px);
        c = kelirDetail(KELIR_D2, 2, c, src, px);
        if (KELIR_LUT_ON != 0) c = kelirColour(c);
    }
    if (KELIR_ROLE != KELIR_ROLE_STAGE_A) c = kelirFinish(c, px);
    kelirOut = vec4(KELIR_WRITES_LINEAR != 0 ? kelirDecode(c) : c, 1.0);
}
