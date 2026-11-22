// SPDX-FileCopyrightText: 2024 Qualcomm Innovation Center, Inc.
// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: BSD-3-Clause
//
// Snapdragon Game Super Resolution 2 (snapdragon-gsr sgsr/v2, 2-pass fragment, convert pass) on a
// game's own frame: motion from the game's velocity buffer, and how much each pixel's depth
// disagrees with its neighbours' (the history there is likely not this surface).
//
// The game's velocity is its own encoding: v = sign(m) sqrt(|m|) / 2 + 127 / 255 per axis, m the
// pixel's move since the frame before in texture coordinates. SGSR 2 takes motion as 2 m.
layout(BINDING(0, 0)) uniform sampler2D depth;
layout(BINDING(1, 1)) uniform sampler2D velocity;
layout(location = 0) out vec4 motionClip;

PARAMS;  // P(0): render size, 1 / render size   P(1): x tan of half the horizontal FOV,
         // y 1 for reversed depth   P(2): x the velocity's zero

vec2 decodeVelocity(vec2 v) {
    vec2 s = (v - P(2).x) * 2.0;
    return sign(s) * s * s;
}

// Distance-ordered depth: near is small, the sky 1.
vec4 ordered(vec4 d) {
    return P(1).y > 0.5 ? 1.0 - d : d;
}

void main() {
    vec2 uv = gl_FragCoord.xy * P(0).zw;
    vec2 gatherCoord = uv - 0.5 * P(0).zw;
    vec4 btmLeft = ordered(textureGather(depth, gatherCoord, 0));
    vec4 btmRight = ordered(textureGather(depth, gatherCoord + vec2(2.0 * P(0).z, 0.0), 0));
    vec4 topLeft = ordered(textureGather(depth, gatherCoord + vec2(0.0, 2.0 * P(0).w), 0));
    vec4 topRight = ordered(textureGather(depth, gatherCoord + 2.0 * P(0).zw, 0));
    float maxC = min(min(min(btmLeft.z, btmRight.w), topLeft.y), topRight.x);

    float depthclip = 0.0;
    if (maxC < 1.0 - 1.0e-05) {
        float btmLeft4 = min(min(min(btmLeft.y, btmLeft.x), btmLeft.z), btmLeft.w);
        float btmRight4 = min(min(min(btmRight.y, btmRight.x), btmRight.z), btmRight.w);
        float topLeft4 = min(min(min(topLeft.y, topLeft.x), topLeft.z), topLeft.w);
        float topRight4 = min(min(min(topRight.y, topRight.x), topRight.z), topRight.w);
        float sep = 1.37e-05 * P(1).x * length(P(0).xy) * (1.0 - maxC);
        const float kEpsilon = 1.19e-07;
        float w = clamp(sep / (abs(maxC - btmLeft4) + kEpsilon), 0.0, 1.0);
        w += clamp(sep / (abs(maxC - btmRight4) + kEpsilon), 0.0, 1.0);
        w += clamp(sep / (abs(maxC - topLeft4) + kEpsilon), 0.0, 1.0);
        w += clamp(sep / (abs(maxC - topRight4) + kEpsilon), 0.0, 1.0);
        depthclip = clamp(1.0 - w * 0.25, 0.0, 1.0);
    }
    vec2 m = decodeVelocity(texelFetch(velocity, ivec2(gl_FragCoord.xy), 0).xy);
    motionClip = vec4(2.0 * m, depthclip, 0.0);
}
