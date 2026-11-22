// Retro: big pixels with few colours and scanlines. Each cell shows the frame at the cell's
// centre after stage A; cells and scanlines are in display px.
// p0: x pixel size (display px), y colour levels, z scanlines

// Stage A's result at a buffer pixel, as the pass that runs Retro can read it.
vec3 kelirRetroTap(ivec2 p);

vec3 kelirRetro(int slot, vec3 c, vec2 duv) {
    vec4 a = kelirScreenParams(slot, 0);
    vec2 display = kelirDisplaySize();
    vec2 cell = floor(duv * display / a.x);
    vec2 centre = kelirBufferUv((cell + 0.5) * a.x / display);
    vec3 s = kelirRetroTap(ivec2(centre * kelirBufferSize()));
    float steps = floor(a.y + 0.5) - 1.0;
    vec3 q = floor(clamp(s, 0.0, 1.0) * steps + 0.5) / steps;
    // The lower part of each row of cells darkens: a scanline per cell row, none under 2 px.
    float within = fract(duv.y * display.y / a.x);
    float scan = 1.0 - a.z * 0.6 * smoothstep(0.45, 0.95, within) * step(2.0, a.x);
    return mix(c, q * scan, kelirScreenIntensity(slot));
}
