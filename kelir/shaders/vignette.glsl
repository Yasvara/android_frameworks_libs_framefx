// Vignette: darkens toward the edges. Roundness 0 follows the display's rectangle, 1 is a circle
// in display pixels reaching the nearer edge; size moves where the falloff starts.
// p0: x size, y roundness
vec3 kelirVignette(int slot, vec3 c, vec2 duv) {
    vec4 a = kelirScreenParams(slot, 0);
    vec2 size = kelirDisplaySize();
    vec2 d = (duv - 0.5) * 2.0;
    vec2 q = mix(d, d * size / min(size.x, size.y), a.y);
    float inner = mix(1.25, 0.15, a.x);
    float v = smoothstep(inner, inner + 0.9, length(q));
    return c * (1.0 - v * kelirScreenIntensity(slot));
}
