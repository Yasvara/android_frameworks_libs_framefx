// Tilt-Shift: blur by place on the display, sharp inside a band (or a circle, the screen-space
// stand-in for depth of field) and growing toward the edges. Distances are in display heights.
// p0: x shape, y position, z size, w softness   p1: x angle (degrees), y blur
vec3 kelirTiltShift(int slot, vec3 c, vec3 src, vec2 px) {
    vec4 a = kelirDetailParams(slot, 0);
    vec4 b = kelirDetailParams(slot, 1);
    vec2 size = kelirDisplaySize();
    vec2 q = (kelirDisplayUv(px) - vec2(0.5, a.y)) * vec2(size.x / size.y, 1.0);
    float dist;
    if (a.x < 0.5) {
        float t = radians(b.x);
        dist = abs(dot(q, vec2(-sin(t), cos(t))));
    } else {
        dist = length(q);
    }
    float edge = 0.5 * a.z;
    float away = smoothstep(edge, edge + 0.5 * max(a.w, 0.01), dist);
    // Full blur reaches the pyramid's last level, 1/64 of the frame; over the first step the
    // frame itself fades into the quarter-resolution level.
    float level = away * b.y * 5.0;
    vec3 blurred = kelirBlur(px / kelirBufferSize(), max(level - 1.0, 0.0));
    float amount = clamp(level, 0.0, 1.0) * kelirDetailIntensity(slot);
    return mix(c, blurred, amount);
}
