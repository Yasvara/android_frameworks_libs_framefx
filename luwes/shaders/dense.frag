// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
//
// The back vector per pixel of half the frame, in frame pixels, for the output pass to warp by:
// the game's own velocity where it renders one Luwes could fit, the block search's field where the
// block matches the two frames well and the game's vector clearly does not (particles,
// transparency and anything else the game leaves out of its velocity) or there is none. Where
// neither matches, the pixel is an occlusion: content the frame before did not show, behind a
// moving character. The game's vector is still its motion there; the block field's is noise.
// The match is a 3x3 luma difference at half size. Alpha carries the game's depth as a distance
// order (0 near, 1 far) for splat.comp, 0.5 without depth.
// The game's vector is never filtered: a blend of a character's and the background's is neither,
// and warps a band along every edge. With depth the pixel takes the vector of the nearest texel
// around it, so the character's edge keeps the character's motion.
layout(BINDING(0, 0)) uniform sampler2D velocity;  // the game's, raw
layout(BINDING(1, 1)) uniform sampler2D field;     // finest block field, its level's pixels
layout(BINDING(2, 2)) uniform sampler2D lumaNow;   // pyramid level 0 of the current frame
layout(BINDING(3, 3)) uniform sampler2D lumaBefore;
layout(BINDING(4, 4)) uniform sampler2D depth;     // the game's, rows as the velocity's
layout(location = 0) out vec4 back;

PARAMS;  // P(0): the game's encoding: x kind (0 none, 1 linear, 2 squared), y bias, zw scale to
         //       frame pixels   P(1): x 1 to read the velocity's rows flipped, y field-to-frame
         //       scale, zw this target's size   P(2): xy frame size / aligned size, z 1 with depth,
         //       w 1 when depth is reversed

vec2 decode(vec2 raw) {
    vec2 v = raw - P(0).y;
    if (P(0).x > 1.5) {
        v *= 2.0;
        v = sign(v) * v * v;
    }
    return v * P(0).zw;
}

// 3x3 luma difference at half size between now at x and before at x + b (b in frame pixels).
float mismatch(vec2 x, vec2 b) {
    vec2 texel = 1.0 / P(1).zw;
    float e = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 o = vec2(float(i), float(j));
            float a = textureLod(lumaNow, (x + o) * texel, 0.0).r;
            float c = textureLod(lumaBefore, (x + o + 0.5 * b) * texel, 0.0).r;
            e += abs(a - c);
        }
    }
    return e;
}

float orderOf(float z) { return P(2).w > 0.5 ? 1.0 - z : z; }

void main() {
    vec2 x = gl_FragCoord.xy;
    vec2 uv = x / P(1).zw;
    vec2 block = textureLod(field, uv, 0.0).xy * P(1).y;
    // The target covers the aligned frame; the game's buffers cover the frame.
    vec2 vuv = uv * P(2).xy;
    if (P(1).x > 0.5) vuv.y = 1.0 - vuv.y;
    ivec2 vsize = textureSize(velocity, 0);
    ivec2 vt = clamp(ivec2(vuv * vec2(vsize)), ivec2(0), vsize - 1);
    float order = 0.5;
    float here = 0.5;
    ivec2 dsize = textureSize(depth, 0);
    if (P(2).z > 0.5) {
        vec2 dscale = vec2(dsize) / vec2(vsize);
        ivec2 dt = clamp(ivec2(vuv * vec2(dsize)), ivec2(0), dsize - 1);
        order = orderOf(texelFetch(depth, dt, 0).r);
        here = order;
        ivec2 best = vt;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                ivec2 o = ivec2(i, j);
                ivec2 dq = clamp(dt + o, ivec2(0), dsize - 1);
                float k = orderOf(texelFetch(depth, dq, 0).r);
                if (k < order) {
                    order = k;
                    best = clamp(ivec2((vec2(dq) + 0.5) / dscale), ivec2(0), vsize - 1);
                }
            }
        }
        vt = best;
    }
    if (P(0).x < 0.5) {
        back = vec4(block, 0.0, order);
        return;
    }
    vec2 game = decode(texelFetch(velocity, vt, 0).xy);
    // Where the content came from, something nearer stands now: it was hidden in the frame
    // before, so no vector matches there and the block field's only matches by chance (a plain
    // floor). The game's vector is the motion.
    bool hidden = false;
    if (P(2).z > 0.5) {
        vec2 frame = 2.0 * P(1).zw * P(2).xy;
        vec2 from = vuv + vec2(game.x, P(1).x > 0.5 ? -game.y : game.y) / frame;
        ivec2 dq = clamp(ivec2(from * vec2(dsize)), ivec2(0), dsize - 1);
        hidden = orderOf(texelFetch(depth, dq, 0).r) < here - 0.01;
    }
    // Mean luma difference under 2 % is a match.
    const float kMatched = 9.0 * 0.02;
    float mb = mismatch(x, block);
    bool useGame = hidden || !(mb < kMatched && mb * 3.0 < mismatch(x, game));
    back = vec4(useGame ? game : block, useGame ? 1.0 : 0.0, order);
}
