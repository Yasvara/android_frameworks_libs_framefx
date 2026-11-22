// Sharpen: contrast-adaptive (AMD FidelityFX CAS), on the rendered frame. Standard takes the
// cross, Plus all nine neighbours. Where the neighbourhood's range is already wide the weight
// falls, so edges never ring. Ignore film grain holds back where the range is only noise, and
// where the game's depth is known it holds back across object edges too.
// p0: x ignore film grain, y quality
vec3 kelirSharpen(int slot, vec3 c, vec3 src, vec2 px) {
    vec4 a = kelirDetailParams(slot, 0);
    ivec2 p = ivec2(px);
    vec3 n = kelirTap(p + ivec2(0, -1));
    vec3 w = kelirTap(p + ivec2(-1, 0));
    vec3 e = kelirTap(p + ivec2(1, 0));
    vec3 s = kelirTap(p + ivec2(0, 1));
    vec3 mn = min(src, min(min(n, w), min(e, s)));
    vec3 mx = max(src, max(max(n, w), max(e, s)));
    vec3 ring = n + w + e + s;
    float count = 4.0;
    vec3 headroom = min(mn, 1.0 - mx);
    if (KELIR_SHARPEN_PLUS != 0) {
        vec3 nw = kelirTap(p + ivec2(-1, -1));
        vec3 ne = kelirTap(p + ivec2(1, -1));
        vec3 sw = kelirTap(p + ivec2(-1, 1));
        vec3 se = kelirTap(p + ivec2(1, 1));
        vec3 mn2 = min(mn, min(min(nw, ne), min(sw, se)));
        vec3 mx2 = max(mx, max(max(nw, ne), max(sw, se)));
        // CAS's softer nine-tap limits: the cross and the full ring, averaged.
        headroom = 0.5 * (min(mn, 1.0 - mx) + min(mn2, 1.0 - mx2));
        mx = mx2;
        mn = mn2;
        ring += 0.5 * (nw + ne + sw + se);
        count = 6.0;
    }
    vec3 amp = sqrt(clamp(headroom / max(mx, vec3(1.0 / 1024.0)), 0.0, 1.0));
    vec3 weight = amp * (-1.0 / 5.0) * (4.0 / count);
    vec3 sharp = (src + ring * weight) / (1.0 + count * weight);
    vec3 delta = sharp - src;
    float range = max(mx.r - mn.r, max(mx.g - mn.g, mx.b - mn.b));
    float floorAt = a.x * 0.08;
    float keep = a.x > 0.0 ? smoothstep(0.5 * floorAt, floorAt, range) : 1.0;
    keep *= 1.0 - kelirEdge(p);
    return c + delta * (keep * kelirDetailIntensity(slot));
}
