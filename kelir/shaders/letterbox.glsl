// Letterbox: frames the display to an aspect ratio, long side to short. Wider than the display
// gives bars across the long edges, narrower gives pillars; intensity is how dark they are.
// p0: x aspect
vec3 kelirLetterbox(int slot, vec3 c, vec2 duv) {
    float target = kelirScreenParams(slot, 0).x;
    vec2 size = kelirDisplaySize();
    bool landscape = size.x >= size.y;
    float longSide = max(size.x, size.y);
    float shortSide = min(size.x, size.y);
    float aspect = longSide / shortSide;
    float t;
    float span;
    float keep;
    if (target >= aspect) {
        t = landscape ? duv.y : duv.x;
        span = shortSide;
        keep = aspect / target;
    } else {
        t = landscape ? duv.x : duv.y;
        span = longSide;
        keep = target / aspect;
    }
    // Display px from the bar's edge, inside positive; the edge pixel is covered by its share.
    float inside = clamp((keep - abs(t - 0.5) * 2.0) * 0.5 * span + 0.5, 0.0, 1.0);
    return c * (1.0 - (1.0 - inside) * kelirScreenIntensity(slot));
}
