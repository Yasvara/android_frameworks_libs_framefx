// Watercolor: washes from the pyramid that wander a little on screen-fixed noise, as pigment
// does on wet paper, lightened toward the paper, with pencil lines from the frame's edges.
// p0: x blurriness, y fuzziness, z pencil detail, w pencil intensity
vec3 kelirWatercolor(int slot, vec3 c, vec3 src, vec2 px) {
    vec4 a = kelirDetailParams(slot, 0);
    vec2 frame = kelirBufferSize();
    vec2 display = kelirDisplaySize();
    vec2 uv = px / frame;
    vec2 dpx = kelirDisplayUv(px) * display;
    // Fuzziness: up to 6 display px of drift in a random direction, constant over 3 px cells.
    vec2 drift = (kelirRandomAt(floor(dpx / 3.0), 0x57a7u).xy - 0.5) * (12.0 * a.y);
    float toFrame = sqrt((frame.x * frame.y) / (display.x * display.y));
    vec3 wash = kelirBlur(uv + drift * toFrame / frame, a.x * 2.0);
    wash = mix(wash, vec3(1.0), 0.12);

    // Pencil: Sobel on the frame's luma; more detail lowers the edge strength a line needs.
    ivec2 p = ivec2(px);
    float tl = kelirLuma(kelirTap(p + ivec2(-1, -1)));
    float t = kelirLuma(kelirTap(p + ivec2(0, -1)));
    float tr = kelirLuma(kelirTap(p + ivec2(1, -1)));
    float l = kelirLuma(kelirTap(p + ivec2(-1, 0)));
    float r = kelirLuma(kelirTap(p + ivec2(1, 0)));
    float bl = kelirLuma(kelirTap(p + ivec2(-1, 1)));
    float b = kelirLuma(kelirTap(p + ivec2(0, 1)));
    float br = kelirLuma(kelirTap(p + ivec2(1, 1)));
    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    float strength = length(vec2(gx, gy));
    float onset = mix(0.9, 0.12, a.z);
    float line = smoothstep(onset, onset + 0.3, strength) * a.w;

    // Paper grain, fixed to the display.
    float paper = 1.0 - 0.05 * kelirRandomAt(floor(dpx / 2.0), 0x9a9eu).x;
    vec3 painting = wash * (1.0 - 0.85 * line) * paper;
    return mix(c, painting, kelirDetailIntensity(slot));
}
