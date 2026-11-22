// Stage B: the screen band in stack order. It runs on every presented frame, fused into the pass
// that writes it, so frame generation never moves grain, dust or bars with the scene.

vec3 kelirScreen(int kind, int slot, vec3 c, vec2 duv) {
#ifdef KELIR_HAS_VIGNETTE
    if (kind == KELIR_VIGNETTE) return kelirVignette(slot, c, duv);
#endif
#ifdef KELIR_HAS_FILM_GRAIN
    if (kind == KELIR_FILM_GRAIN) return kelirFilmGrain(slot, c, duv);
#endif
#ifdef KELIR_HAS_LETTERBOX
    if (kind == KELIR_LETTERBOX) return kelirLetterbox(slot, c, duv);
#endif
#ifdef KELIR_HAS_OLD_FILM
    if (kind == KELIR_OLD_FILM) return kelirOldFilm(slot, c, duv);
#endif
#ifdef KELIR_HAS_RETRO
    if (kind == KELIR_RETRO) return kelirRetro(slot, c, duv);
#endif
#ifdef KELIR_HAS_HALFTONE
    if (kind == KELIR_HALFTONE) return kelirHalftone(slot, c, duv);
#endif
    return c;
}

vec3 kelir_stage_b(vec3 c, vec2 duv) {
    c = kelirScreen(KELIR_S0, 0, c, duv);
    c = kelirScreen(KELIR_S1, 1, c, duv);
    c = kelirScreen(KELIR_S2, 2, c, duv);
    c = kelirScreen(KELIR_S3, 3, c, duv);
    return c;
}
