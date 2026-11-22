// Details: clarity, HDR toning and bloom from the shared pyramid, as Freestyle's Details filter
// has them (its Sharpen slider is the Sharpen kind).
// p0: x clarity, y HDR toning, z bloom, w bloom threshold

// The part of a blurred colour above the threshold, with a soft knee so highlights fade in.
vec3 kelirBright(vec3 c, float threshold) {
    float peak = max(c.r, max(c.g, c.b));
    float knee = 0.05 + 0.25 * (1.0 - threshold);
    float w = clamp((peak - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    return c * (w * w);
}

vec3 kelirDetails(int slot, vec3 c, vec3 src, vec2 px) {
    vec4 a = kelirDetailParams(slot, 0);
    vec2 uv = px / kelirBufferSize();
    float l = kelirLuma(src);
    float lift = 0.0;
    if (a.x != 0.0) {
        // Local contrast against a 1/16 blur, on luma so edges keep their colour; faded toward
        // black and white so it never clips them.
        float base = kelirLuma(kelirBlur(uv, 2.0));
        float mid = clamp(4.0 * l * (1.0 - l), 0.0, 1.0);
        lift += a.x * (l - base) * mid;
    }
    if (a.y != 0.0) {
        // HDR toning: the local mean pulled toward middle grey, the detail around it lifted.
        float mean = kelirLuma(kelirBlur(uv, 3.0));
        lift += a.y * (0.5 * (0.5 - mean) + 0.35 * (l - mean));
    }
    vec3 glow = vec3(0.0);
    if (a.z != 0.0) {
        glow = 0.30 * kelirBright(kelirBlur(uv, 1.0), a.w) + 0.30 * kelirBright(kelirBlur(uv, 2.0), a.w) +
                0.25 * kelirBright(kelirBlur(uv, 3.0), a.w) + 0.15 * kelirBright(kelirBlur(uv, 4.0), a.w);
        glow *= a.z;
    }
    return c + (vec3(lift) + glow) * kelirDetailIntensity(slot);
}
