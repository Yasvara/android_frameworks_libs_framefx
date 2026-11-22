// Writes one generated frame at phase t.
//
// The field holds back vectors b per block of the finest pyramid level: content at q in the
// current frame sat at q + b in the previous one. Both modes gather, solving for the source by
// fixed-point iteration so foreground edges follow the foreground:
//   interpolate, t from previous to current: q = p - b(q) * (1 - t), blend current at q with
//     previous at q + b(q);
//   extrapolate, t past the current frame: q = p + b(q) * t, current at q.
// Where the two frames disagree about what moved there the vector is wrong (occlusion, motion the
// search did not find), and the pixel falls back toward the plain cross-fade, or toward the real
// pixel when extrapolating: a soft frame instead of a torn one. Where the vector is the game's own
// it is right, and a disagreement is an occlusion: the pixel takes one frame's warped sample, so
// a moving character leaves no second copy behind. Pixels the two frames agree on
// stay exactly as they are, which keeps HUD and static scenery still.
layout(BINDING(0, 0)) uniform sampler2D current;
layout(BINDING(1, 1)) uniform sampler2D previous;
layout(BINDING(2, 2)) uniform sampler2D field;
// With P(2).x: the HUD layer and mask of the real frame `current` carries.
layout(BINDING(3, 3)) uniform sampler2D hudLayer;
layout(BINDING(4, 4)) uniform sampler2D hudMask;
layout(location = 0) out vec4 color;

PARAMS;  // P(0): frame size, aligned size   P(1): x t, y mode, z field-to-frame scale
         // P(2): x 1 when the frames carry the HUD of bindings 3 and 4, y 1 to leave the HUD out
         //       of a composed frame, z 1 when the frames are the 3D alone and the HUD of
         //       bindings 3 and 4 goes over the result, w 1 when the field is per pixel in frame
         //       pixels with z 1 where the vector is the game's   P(3): x 1 when the field is
         //       already at phase t (splat.comp): each pixel holds the vector of what is there,
         //       alpha 2 where the current frame covers it

// A frame the HUD is over is (1 - h) X + layer: X back, where the HUD lets any of it through.
vec3 under(vec3 c, ivec2 q) {
    vec3 layer = texelFetch(hudLayer, q, 0).rgb * 2.0 - 1.0;
    float h = texelFetch(hudMask, q, 0).r;
    return clamp((c - layer) / max(1.0 - h, 1e-3), 0.0, 1.0);
}

// With Kelir, every frame this pass writes gets Kelir's finish: stage B on the 3D under the HUD,
// then the divider and dither over all. Retro reads the held real frame, which already carries
// stage A.
#ifdef KELIR_FINISH
vec3 kelirRetroTap(ivec2 p) {
    vec3 c = texelFetch(current, p, 0).rgb;
    return P(2).x > 0.5 ? under(c, p) : c;
}

vec3 finished(vec3 c, vec2 px) {
    if (P(2).x > 0.5) {
        ivec2 q = ivec2(px);
        vec3 layer = texelFetch(hudLayer, q, 0).rgb * 2.0 - 1.0;
        float h = texelFetch(hudMask, q, 0).r;
        c = clamp((1.0 - h) * kelirUnder(under(c, q), px) + layer, 0.0, 1.0);
    } else {
        c = kelirUnder(c, px);
    }
    return kelirOver(c, px);
}
#else
#define kelirUnder(c, px) (c)
#define kelirOver(c, px) (c)
#define finished(c, px) (c)
#endif

const int kExtrapolate = 0;
const int kInterpolate = 1;
const int kComposeLayer = 3;

vec2 backAt(vec2 q) {
    return texture(field, (q + 0.5) / P(0).zw).xy * P(1).z;
}

// At phase t the field is per half-size pixel, and a filtered read across a character's edge
// blends its vector with the background's, which warps a band around it. Where the four vectors
// around p differ, p takes the one of the texel it lies in.
vec2 phaseBackAt(vec2 p) {
    vec2 f = (p + 0.5) / P(0).zw * vec2(textureSize(field, 0)) - 0.5;
    ivec2 o = ivec2(floor(f));
    ivec2 last = textureSize(field, 0) - 1;
    vec2 lo = vec2(1e9);
    vec2 hi = vec2(-1e9);
    for (int i = 0; i < 4; i++) {
        vec2 v = texelFetch(field, clamp(o + ivec2(i & 1, i >> 1), ivec2(0), last), 0).xy;
        lo = min(lo, v);
        hi = max(hi, v);
    }
    vec2 spread = (hi - lo) * P(1).z;
    if (max(spread.x, spread.y) < 1.0) return backAt(p);
    return texelFetch(field, clamp(ivec2(floor(f + 0.5)), ivec2(0), last), 0).xy * P(1).z;
}

// How much the content at p (phase field) is covered in the current frame.
float coveredAt(vec2 p) {
    return clamp(texture(field, (p + 0.5) / P(0).zw).w - 1.0, 0.0, 1.0);
}

// How much the vector at q is the game's own.
float gameAt(vec2 q) {
    return P(2).w > 0.5 ? texture(field, (q + 0.5) / P(0).zw).z : 0.0;
}

vec3 sampleAt(sampler2D frame, vec2 q) {
    return texture(frame, (q + 0.5) / P(0).xy).rgb;
}

float disagreement(vec3 a, vec3 b) {
    vec3 d = abs(a - b);
    return clamp((0.5 * d.r + d.g + 0.5 * d.b - 0.01) * 8.0, 0.0, 1.0);
}

// current: the game's 3D under its HUD, upscaled and graded, or a frame generated from it;
// previous: the HUD layer hud.frag stored; field: its mask. P(2).y: 1 leaves the HUD out (photo
// mode).
vec3 composeLayer(ivec2 q, vec2 px) {
    vec3 x = kelirUnder(texelFetch(current, q, 0).rgb, px);
    if (P(2).y > 0.5) return x;
    vec3 layer = texelFetch(previous, q, 0).rgb * 2.0 - 1.0;
    float h = texelFetch(field, q, 0).r;
    return clamp((1.0 - h) * x + layer, 0.0, 1.0);
}

// A generated pixel as written: over the 3D alone, this frame's HUD goes on top.
vec3 emit(vec3 c, vec2 px) {
    if (P(2).z < 0.5) return finished(c, px);
    vec3 x = kelirUnder(c, px);
    if (P(2).y > 0.5) return kelirOver(x, px);
    ivec2 q = ivec2(px);
    vec3 layer = texelFetch(hudLayer, q, 0).rgb * 2.0 - 1.0;
    float h = texelFetch(hudMask, q, 0).r;
    return kelirOver(clamp((1.0 - h) * x + layer, 0.0, 1.0), px);
}

void main() {
    vec2 p = floor(gl_FragCoord.xy);
    if (int(P(1).y) == kComposeLayer) {
        color = vec4(kelirOver(composeLayer(ivec2(p), gl_FragCoord.xy), gl_FragCoord.xy), 1.0);
        return;
    }
    vec3 now = texelFetch(current, ivec2(p), 0).rgb;
    vec3 before = texelFetch(previous, ivec2(p), 0).rgb;
    vec3 d = abs(now - before);
    if (max(d.r, max(d.g, d.b)) < 2.0 / 255.0) {
        color = vec4(emit(now, gl_FragCoord.xy), 1.0);
        return;
    }
    float t = P(1).x;
    vec3 result;
    bool atPhase = P(3).x > 0.5;
    if (int(P(1).y) == kInterpolate) {
        vec2 q = p;
        vec2 bp = atPhase ? phaseBackAt(p) : vec2(0.0);
        if (atPhase) {
            q = p - bp * (1.0 - t);
        } else {
            for (int i = 0; i < 3; i++) q = p - backAt(q) * (1.0 - t);
        }
        vec3 c1 = sampleAt(current, q);
        vec2 bq = atPhase ? bp : backAt(q);
        vec3 c0 = sampleAt(previous, q + bq);
        vec3 warped = mix(c0, c1, t);
        float g = atPhase ? gameAt(p) : gameAt(q);
        // At phase the pixel's content was splatted from the current frame, so c1 is it and c0
        // is what the frame before had there, maybe something in front of it.
        vec3 fallback = mix(mix(before, now, t), atPhase || t >= 0.5 ? c1 : c0, g);
        result = mix(warped, fallback, disagreement(c0, c1) * mix(0.5, 1.0, g));
        // Covered now: only the previous frame shows it.
        if (atPhase) result = mix(result, c0, coveredAt(p));
    } else {
        vec2 q = p;
        vec2 bp = atPhase ? phaseBackAt(p) : vec2(0.0);
        if (atPhase) {
            q = p + bp * t;
        } else {
            for (int i = 0; i < 3; i++) q = p + backAt(q) * t;
        }
        vec2 b = atPhase ? bp : backAt(q);
        vec3 c1 = sampleAt(current, q);
        vec3 c0 = sampleAt(previous, q + b);
        float g = atPhase ? gameAt(p) : gameAt(q);
        result = mix(c1, now, disagreement(c0, c1) * 0.5 * (1.0 - g));
        if (atPhase) result = mix(result, c0, coveredAt(p));
    }
    color = vec4(emit(result, gl_FragCoord.xy), 1.0);
}
