// Painterly: a generalised Kuwahara filter on the quarter-resolution level. Four overlapping
// sectors around the pixel each give a mean and a variance; flat sectors win, so strokes follow
// edges instead of blurring across them. Detail lays the frame's own fine structure back in as
// brush texture.
// p0: x brush, y detail
vec3 kelirPainterly(int slot, vec3 c, vec3 src, vec2 px) {
    vec4 a = kelirDetailParams(slot, 0);
    vec2 uv = px / kelirBufferSize();
    vec2 texel = 1.0 / vec2(textureSize(kelirPyramid, 0));
    float reach = 0.75 + 1.75 * a.x;
    vec3 sum = vec3(0.0);
    float total = 0.0;
    for (int s = 0; s < 4; s++) {
        vec2 dir = vec2(s == 1 || s == 3 ? 1.0 : -1.0, s >= 2 ? 1.0 : -1.0) * texel * reach;
        vec3 mean = vec3(0.0);
        float m1 = 0.0;
        float m2 = 0.0;
        for (int t = 0; t < 4; t++) {
            vec2 at = vec2(t == 1 || t == 3 ? 1.5 : 0.5, t >= 2 ? 1.5 : 0.5);
            vec3 v = kelirBlur(uv + dir * at, 0.0);
            float y = kelirLuma(v);
            mean += v;
            m1 += y;
            m2 += y * y;
        }
        mean *= 0.25;
        float variance = max(m2 * 0.25 - m1 * m1 * 0.0625, 0.0);
        float w = 1.0 / (1e-4 + variance * variance * 1e4);
        sum += mean * w;
        total += w;
    }
    vec3 paint = sum / total;
    paint += a.y * 0.6 * (src - kelirBlur(uv, 0.0));
    return mix(c, paint, kelirDetailIntensity(slot));
}
