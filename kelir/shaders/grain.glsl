// Film Grain: a new field every present, in grains of a size in display px, heaviest in the
// midtones as on film. Each grain is triangular noise in [-1, 1]; Colour gives each channel its
// own.
// p0: x size (display px), y colour
vec3 kelirFilmGrain(int slot, vec3 c, vec2 duv) {
    vec4 a = kelirScreenParams(slot, 0);
    vec2 cell = floor(duv * kelirDisplaySize() / a.x);
    vec3 u = kelirRandomAt(cell, kelirSeed());
    vec3 v = kelirRandomAt(cell, kelirSeed() ^ 0x6a09e667u);
    vec3 noise = mix(vec3(u.x + v.x - 1.0), u + v - 1.0, a.y);
    float l = clamp(kelirLuma(c), 0.0, 1.0);
    float amplitude = 0.10 * (0.25 + 3.0 * l * (1.0 - l));
    return c + noise * (amplitude * kelirScreenIntensity(slot));
}
