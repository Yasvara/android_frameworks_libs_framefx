// SPDX-FileCopyrightText: 2024 Qualcomm Innovation Center, Inc.
// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: BSD-3-Clause
//
// Snapdragon Game Super Resolution 2 (snapdragon-gsr sgsr/v2, 2-pass fragment, upscale pass) on the
// game's jittered scene before its own anti-aliasing, accumulated at the output size; then what
// that adds over the game's own resolve, carried onto the game's finished scene.
//
// The game's post-processing (bloom, tonemap, grading) runs at its render size on its resolve R,
// and gives S. Locally S follows R as a power, so the finished scene at the output size is
// S stretched times (hi / R stretched)^gamma, hi the accumulation: where SGSR 2 finds no more
// than the game did, the game's own stretch comes out unchanged.
//
// Everything is computed in the game's texture space; the output is the frame's, so coordinates
// go through P(2).zw's flips. History is kept in the frame's orientation, in the reversible
// tonemap c / (1 + max(c)) the colour box needs.
layout(BINDING(0, 0)) uniform sampler2D history;     // linear
layout(BINDING(1, 1)) uniform sampler2D motionClip;  // linear
layout(BINDING(2, 2)) uniform sampler2D jittered;    // texel fetches
layout(BINDING(3, 3)) uniform sampler2D base;        // S, or Kelir's scene band over it; linear
layout(BINDING(4, 4)) uniform sampler2D resolved;    // R; linear
layout(location = 0) out vec4 accumulated;
layout(location = 1) out vec4 scene;

PARAMS8;  // P(0): render size, 1 / render size   P(1): output size, 1 / output size
          // P(2): xy jitter (render pixels), zw x and y flipped   P(3): x kernel scale, y box
          // scale, z min lerp contribution, w reset   P(4): x gamma, y largest detail gain,
          // z weight of a new sample against history

float fastLanczos(float base) {
    float y = base - 1.0;
    float y2 = y * y;
    float t = 0.75 * y + y2;
    return t * y2;
}

vec3 sanitizeHdr(vec3 c) {
    vec3 clamped = clamp(c, vec3(0.0), vec3(65504.0));
    return mix(clamped, vec3(0.0), isnan(c));
}

vec3 tonemap(vec3 c) {
    c = sanitizeHdr(c);
    return c / (1.0 + max(c.r, max(c.g, c.b)));
}

vec3 untonemap(vec3 c) {
    c = mix(clamp(c, vec3(0.0), vec3(0.999)), vec3(0.0), isnan(c));
    return c / max(1.0 - max(c.r, max(c.g, c.b)), 1e-3);
}

vec2 flipped(vec2 uv) {
    return mix(uv, 1.0 - uv, P(2).zw);
}

void main() {
    vec2 frameUv = gl_FragCoord.xy * P(1).zw;
    vec2 hruv = flipped(frameUv);
    vec2 renderSize = P(0).xy;
    vec2 jitter = P(2).xy;

    vec2 jitterUv = clamp(hruv + jitter * P(0).zw, 0.0, 1.0);
    ivec2 last = max(ivec2(renderSize) - 1, ivec2(0));
    ivec2 inputPos = clamp(ivec2(jitterUv * renderSize), ivec2(0), last);
    vec3 mda = textureLod(motionClip, jitterUv, 0.0).xyz;
    vec2 motion = mda.xy;
    vec2 prevUv = clamp(hruv - 0.5 * motion, 0.0, 1.0);
    float depthfactor = mda.z;
    vec3 historyColor = textureLod(history, flipped(prevUv), 0.0).rgb;

    float biasmax = P(3).x;
    float biasmin = max(1.0, 0.3 + 0.3 * biasmax);
    float kernelbias = mix(biasmax, biasmin, 0.25 * depthfactor) * 0.5;
    float kernelbias2 = kernelbias * kernelbias;
    float motionLen = length(motion * P(1).xy);
    float curvebias = mix(-2.0, -3.0, clamp(motionLen * 0.02, 0.0, 1.0));

    vec2 offset = vec2(inputPos) + 0.5 - jitter - hruv * renderSize;
    vec4 upsampled = vec4(0.0);
    vec3 centerTap = vec3(0.0);
    vec3 boxCenter = vec3(0.0);
    vec3 boxVar = vec3(0.0);
    float boxWeight = 0.0;
    vec3 boxMin = vec3(1e9);
    vec3 boxMax = vec3(-1e9);
    const ivec2 kTaps[5] = ivec2[5](ivec2(0, 1), ivec2(1, 0), ivec2(-1, 0), ivec2(0, 0), ivec2(0, -1));
    for (int k = 0; k < 5; k++) {
        vec3 c = tonemap(texelFetch(jittered, clamp(inputPos + kTaps[k], ivec2(0), last), 0).rgb);
        if (k == 3) centerTap = c;
        vec2 o = offset + vec2(kTaps[k]);
        float d = dot(o, o);
        float weight = fastLanczos(clamp(d * kernelbias2, 0.0, 1.0));
        upsampled += vec4(c * weight, weight);
        float bw = exp(d * curvebias);
        boxMin = min(boxMin, c);
        boxMax = max(boxMax, c);
        boxCenter += c * bw;
        boxVar += c * c * bw;
        boxWeight += bw;
    }
    float invBoxW = 1.0 / max(boxWeight, 1e-6);
    boxCenter *= invBoxW;
    boxVar = sqrt(max(boxVar * invBoxW - boxCenter * boxCenter, vec3(0.0)));
    upsampled.rgb = upsampled.w > 1e-5
            ? clamp(upsampled.rgb / upsampled.w, boxMin - 0.075, boxMax + 0.075)
            : centerTap;
    historyColor = mix(clamp(historyColor, vec3(0.0), vec3(0.999)), upsampled.rgb,
                       isnan(historyColor));
    // Lower than SGSR 2's own third: games dither transparency and leave it to their
    // anti-aliasing to average, which takes more frames than SGSR 2 keeps.
    upsampled.w *= P(4).z;

    float baseupdate = 1.0 - depthfactor;
    baseupdate = min(baseupdate, mix(baseupdate, upsampled.w * 10.0, clamp(10.0 * motionLen, 0.0, 1.0)));
    baseupdate = min(baseupdate, mix(baseupdate, upsampled.w, clamp(motionLen * 0.05, 0.0, 1.0)));
    float basealpha = baseupdate;

    float boxscale = max(depthfactor, clamp(motionLen * 0.05, 0.0, 1.0));
    vec3 sboxvar = boxVar * mix(P(3).y, 1.0, boxscale);
    boxMax = min(boxMax, boxCenter + sboxvar);
    boxMin = max(boxMin, boxCenter - sboxvar);
    vec3 clamped = clamp(historyColor, boxMin, boxMax);
    float startLerp = abs(mda.x) + abs(mda.y) > 0.000001 ? 0.0 : P(3).z;
    bool outside = any(greaterThan(boxMin, historyColor)) || any(greaterThan(historyColor, boxMax));
    float lerpContribution = outside ? startLerp : 1.0;
    historyColor = mix(clamped, historyColor, lerpContribution);
    basealpha = mix(min(basealpha, 0.1), basealpha, lerpContribution);
    float alpha = clamp(upsampled.w / max(1.192e-07, basealpha + upsampled.w) + P(3).w, 0.0, 1.0);
    vec3 hi = mix(historyColor, upsampled.rgb, alpha);
    accumulated = vec4(hi, 0.0);

    const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
    vec4 s = textureLod(base, hruv, 0.0);
    float yHi = max(dot(untonemap(hi), kLuma), 0.0);
    float yR = max(dot(sanitizeHdr(textureLod(resolved, hruv, 0.0).rgb), kLuma), 0.0);
    float eps = 1e-3 + 0.01 * yR;
    float gain = clamp(pow((yHi + eps) / (yR + eps), P(4).x), 1.0 / P(4).y, P(4).y);
    scene = vec4(clamp(s.rgb * gain, 0.0, 1.0), s.a);
}
