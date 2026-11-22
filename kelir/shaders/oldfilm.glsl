// Old Film: aged monochrome with dust, scratches and flicker, all new every present. Dust and
// scratches are placed in display px so they look the same at any render scale.
// p0: x dust, y scratches, z flicker
vec3 kelirOldFilm(int slot, vec3 c, vec2 duv) {
    vec4 a = kelirScreenParams(slot, 0);
    uint seed = kelirSeed();
    vec2 dpx = duv * kelirDisplaySize();
    vec3 film = vec3(kelirLuma(c)) * vec3(1.0, 0.96, 0.88);
    film *= 1.0 + (kelirRandom(uvec3(seed, 0xf11cu, 0u)).x - 0.5) * 0.16 * a.z;

    // Dust: at most one speck per 8 px cell, 0.6 to 2.2 px across.
    vec2 cell = floor(dpx / 8.0);
    vec3 h = kelirRandomAt(cell, seed ^ 0xd057u);
    float there = step(1.0 - 0.03 * a.x, kelirRandomAt(cell, seed ^ 0xd058u).x);
    float radius = 0.6 + 1.6 * h.z;
    float speck = there * (1.0 - smoothstep(radius - 0.5, radius + 0.5, length(dpx - (cell + h.xy) * 8.0)));
    film = mix(film, vec3(0.06), speck);

    // Scratches: up to three bright vertical lines, 1 px wide.
    for (int i = 0; i < 3; i++) {
        vec3 s = kelirRandom(uvec3(seed, 0x5c7au, uint(i)));
        float x = s.x * kelirDisplaySize().x;
        float line = step(s.y, 0.6 * a.y) * (1.0 - smoothstep(0.5, 1.5, abs(dpx.x - x)));
        film = mix(film, vec3(0.9), line * (0.35 + 0.4 * s.z));
    }
    return mix(c, film, kelirScreenIntensity(slot));
}
