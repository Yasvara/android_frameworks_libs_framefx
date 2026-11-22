// The last touches of every presented frame, real or generated: stage B, then the A/B divider and
// a dither of half an LSB at the target's depth. Left of the divider the frame stays as it came.
// Generation never sees any of it: it is screen-fixed, and runs after.

// Interleaved gradient noise (Jimenez) at a display pixel, in [-0.5, 0.5).
float kelirDither(vec2 dpx) {
    return fract(52.9829189 * fract(dot(floor(dpx), vec2(0.06711056, 0.00583715)))) - 0.5;
}

// Stage B right of the divider. Where Luwes composes the game's HUD it runs on the 3D alone.
vec3 kelirUnder(vec3 c, vec2 px) {
    vec2 duv = kelirDisplayUv(px);
    return duv.x >= FRAME(1).z ? kelir_stage_b(c, duv) : c;
}

// The dither and the divider, over everything.
vec3 kelirOver(vec3 c, vec2 px) {
    vec2 duv = kelirDisplayUv(px);
    vec2 dpx = duv * kelirDisplaySize();
    float split = FRAME(1).z;
    if (duv.x >= split) c += kelirDither(dpx) * FRAME(2).w * 2.0;
    if (split >= 0.0 && abs(dpx.x - split * kelirDisplaySize().x) < 0.5) c = mix(c, vec3(1.0), 0.75);
    return c;
}

vec3 kelirFinish(vec3 c, vec2 px) {
    return kelirOver(kelirUnder(c, px), px);
}
