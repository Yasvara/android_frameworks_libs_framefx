// SPDX-FileCopyrightText: 2025 Qualcomm Innovation Center, Inc.
// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: BSD-3-Clause
//
// Snapdragon Game Super Resolution 1 (snapdragon-gsr sgsr/v1, sgsr1_shader_mobile.frag, RGBA
// mode): the game's scene before its HUD, upscaled to the swapchain with a 12-tap Lanczos-like
// filter that sharpens edges and leaves flat areas bilinear. Plain mode is the bilinear alone, for
// a scene that is not upscaled; downsampling is Lanczos 2 over every source texel the output
// pixel's kernel covers, for a game drawn larger than the display. Alpha passes through.
layout(BINDING(0, 0)) uniform sampler2D source;
layout(location = 0) out vec4 color;

PARAMS;  // P(0): 1 / source size, source size   P(1): output size, x and y flipped (0 or 1)
         // P(2): x 0 SGSR 1, 1 plain, 2 downsample, y SGSR 1's edge sharpness, 0 its own

const float kEdgeThreshold = 8.0 / 255.0;
const float kEdgeSharpness = 2.0;  // the filter's own, where none is asked for
const int kLuma = 1;  // green stands in for luma

float fastLanczos2(float x) {
    float wA = x - 4.0;
    float wB = x * wA - wA;
    wA *= wA;
    return wB * wA;
}

vec2 weightY(float dx, float dy, float c, float stdev) {
    float x = (dx * dx + dy * dy) * 0.55 + clamp(abs(c) * stdev, 0.0, 1.0);
    float w = fastLanczos2(x);
    return vec2(w, w * c);
}

float lanczos2(float x) {
    x = abs(x);
    if (x < 1e-4) return 1.0;
    if (x >= 2.0) return 0.0;
    const float kPi = 3.14159265;
    return 2.0 * sin(kPi * x) * sin(0.5 * kPi * x) / (kPi * kPi * x * x);
}

vec4 downsample(vec2 uv) {
    vec2 ratio = P(0).zw / P(1).xy;  // source texels per output pixel
    vec2 center = uv * P(0).zw - 0.5;
    vec2 radius = 2.0 * ratio;
    ivec2 lo = ivec2(ceil(center - radius));
    ivec2 hi = ivec2(floor(center + radius));
    ivec2 last = ivec2(P(0).zw) - 1;
    vec4 sum = vec4(0.0);
    float total = 0.0;
    for (int y = lo.y; y <= hi.y; y++) {
        float wy = lanczos2((float(y) - center.y) / ratio.y);
        for (int x = lo.x; x <= hi.x; x++) {
            float w = wy * lanczos2((float(x) - center.x) / ratio.x);
            sum += w * texelFetch(source, clamp(ivec2(x, y), ivec2(0), last), 0);
            total += w;
        }
    }
    return clamp(sum / total, 0.0, 1.0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / P(1).xy;
    uv = mix(uv, 1.0 - uv, P(1).zw);
    if (P(2).x > 1.5) {
        color = downsample(uv);
        return;
    }
    vec4 base = textureLod(source, uv, 0.0);
    if (P(2).x > 0.5) {
        color = base;
        return;
    }
    vec4 c = vec4(base.rgb, 0.0);

    vec2 imgCoord = uv * P(0).zw + vec2(-0.5, 0.5);
    vec2 imgCoordPixel = floor(imgCoord);
    vec2 coord = imgCoordPixel * P(0).xy;
    vec2 pl = imgCoord - imgCoordPixel;
    vec4 left = textureGather(source, coord, kLuma);

    float edgeVote = abs(left.z - left.y) + abs(c[kLuma] - left.y) + abs(c[kLuma] - left.z);
    if (edgeVote > kEdgeThreshold) {
        coord.x += P(0).x;
        vec4 right = textureGather(source, coord + vec2(P(0).x, 0.0), kLuma);
        vec4 upDown;
        upDown.xy = textureGather(source, coord + vec2(0.0, -P(0).y), kLuma).wz;
        upDown.zw = textureGather(source, coord + vec2(0.0, P(0).y), kLuma).yx;

        float mean = (left.y + left.z + right.x + right.w) * 0.25;
        left -= vec4(mean);
        right -= vec4(mean);
        upDown -= vec4(mean);
        c.w = c[kLuma] - mean;

        float sum = abs(left.x) + abs(left.y) + abs(left.z) + abs(left.w) + abs(right.x) +
                abs(right.y) + abs(right.z) + abs(right.w) + abs(upDown.x) + abs(upDown.y) +
                abs(upDown.z) + abs(upDown.w);
        float stdev = 2.181818 / sum;

        vec2 aWY = weightY(pl.x, pl.y + 1.0, upDown.x, stdev);
        aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, stdev);
        aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, stdev);
        aWY += weightY(pl.x, pl.y - 2.0, upDown.w, stdev);
        aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x, stdev);
        aWY += weightY(pl.x, pl.y - 1.0, left.y, stdev);
        aWY += weightY(pl.x, pl.y, left.z, stdev);
        aWY += weightY(pl.x + 1.0, pl.y, left.w, stdev);
        aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x, stdev);
        aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y, stdev);
        aWY += weightY(pl.x - 2.0, pl.y, right.z, stdev);
        aWY += weightY(pl.x - 1.0, pl.y, right.w, stdev);

        float finalY = aWY.y / aWY.x;
        float maxY = max(max(left.y, left.z), max(right.x, right.w));
        float minY = min(min(left.y, left.z), min(right.x, right.w));
        finalY = clamp((P(2).y > 0.0 ? P(2).y : kEdgeSharpness) * finalY, minY, maxY);
        // Smooths high-contrast input.
        float deltaY = clamp(finalY - c.w, -23.0 / 255.0, 23.0 / 255.0);
        c.rgb = clamp(c.rgb + deltaY, 0.0, 1.0);
    }
    color = vec4(c.rgb, base.a);
}
