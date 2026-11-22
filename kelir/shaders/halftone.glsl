// Halftone: printed dots on a screen of a size in display px. Mono prints black on white at 45
// degrees; CMYK prints each ink on its own screen angle and multiplies them on the paper.
// p0: x dot size (display px), y ink

// How much of the paper the dot covers at p for an ink coverage: a disc of that area per cell,
// antialiased over one display px.
float kelirDot(vec2 p, float size, float degrees, float coverage) {
    float t = radians(degrees);
    vec2 r = vec2(cos(t) * p.x + sin(t) * p.y, -sin(t) * p.x + cos(t) * p.y) / size;
    float d = length(fract(r) - 0.5);
    float radius = sqrt(clamp(coverage, 0.0, 1.0) / 3.14159265);
    float aa = 0.5 / size;
    return 1.0 - smoothstep(radius - aa, radius + aa, d);
}

vec3 kelirHalftone(int slot, vec3 c, vec2 duv) {
    vec4 a = kelirScreenParams(slot, 0);
    vec2 p = duv * kelirDisplaySize();
    vec3 cc = clamp(c, 0.0, 1.0);
    vec3 printed;
    if (a.y < 0.5) {
        printed = vec3(1.0 - kelirDot(p, a.x, 45.0, 1.0 - kelirLuma(cc)));
    } else {
        float k = 1.0 - max(cc.r, max(cc.g, cc.b));
        vec3 cmy = (1.0 - cc - k) / max(1.0 - k, 1e-4);
        float cyan = kelirDot(p, a.x, 15.0, cmy.x);
        float magenta = kelirDot(p, a.x, 75.0, cmy.y);
        float yellow = kelirDot(p, a.x, 0.0, cmy.z);
        float black = kelirDot(p, a.x, 45.0, k);
        printed = (1.0 - vec3(cyan, magenta, yellow)) * (1.0 - black);
    }
    return mix(c, printed, kelirScreenIntensity(slot));
}
