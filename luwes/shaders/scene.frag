// Kelir's scene band on the game's 3D before its HUD, at the scene's size. Mode 0: ambient
// occlusion, global illumination, then depth fog. Mode 1: depth of field. Mode 2, the look: edge
// smoothing, sky, sun shafts, tonemap, then outlines. All write in alpha how far the pixel sits on
// a depth edge, where Kelir's Sharpen holds back.
layout(BINDING(0, 0)) uniform sampler2D source;
layout(BINDING(1, 1)) uniform sampler2D lin;
layout(BINDING(2, 2)) uniform sampler2D occlusion;  // half size
layout(BINDING(3, 3)) uniform sampler2D focus;      // 1 x 1; the sun (sun.frag) in mode 2
layout(BINDING(4, 4)) uniform sampler2D bounced;    // quarter size, linear light
// Ray-traced kinds (rays.comp), half size: x sun, y occlusion; reflection premultiplied with its
// weight; bounced light, linear.
layout(BINDING(5, 5)) uniform sampler2D rtVisibility;
layout(BINDING(6, 6)) uniform sampler2D rtReflection;
layout(BINDING(7, 7)) uniform sampler2D rtBounce;
layout(location = 0) out vec4 color;

// P(0): scene size, 1 / size   P(1)..P(6): kelir::PackScene, or kelir::PackSceneLook in mode 2
// P(7): x mode, y 1 when this frame traced rays
PARAMS8;

const float kSky = 50000.0;
const int kTaps = 16;

float depthAt(vec2 uv) {
    return textureLod(lin, uv, 0.0).r;
}

float edgeAt(vec2 uv, float z) {
    vec2 tx = P(0).zw;
    float l = depthAt(uv - vec2(tx.x, 0.0));
    float r = depthAt(uv + vec2(tx.x, 0.0));
    float b = depthAt(uv - vec2(0.0, tx.y));
    float t = depthAt(uv + vec2(0.0, tx.y));
    float jump = max(max(abs(l - z), abs(r - z)), max(abs(b - z), abs(t - z)));
    return smoothstep(0.02, 0.08, jump / max(z, 1e-4));
}

// The half-size occlusion at full size: the four nearest half-size pixels and the one under the
// pixel, each counted as much as its depth agrees with the pixel's.
float bilateralWeight(float dz, float invScale) {
    float r = dz * invScale;
    return 1.0 / (1.0 + r * r);
}

float occlusionAt(vec2 uv, float z) {
    vec2 cell = 2.0 * P(0).zw;
    float invScale = 1.0 / max(0.05 * z, 1e-4);
    float sum = textureLod(occlusion, uv, 0.0).r;
    float weight = 1.0;
    for (int i = 0; i < 4; i++) {
        vec2 o = vec2(i == 0 || i == 1 ? -1.0 : 1.0, i == 0 || i == 2 ? -1.0 : 1.0) * cell;
        float w = bilateralWeight(depthAt(uv + o) - z, invScale);
        sum += textureLod(occlusion, uv + o, 0.0).r * w;
        weight += w;
    }
    return sum / weight;
}

// A half-size ray-traced result at full size: the nine nearest half-size pixels, each counted as
// much as its depth agrees with the pixel's, so one ray a pixel reads as a soft, edge-true field.
vec4 tracedAt(sampler2D image, vec2 uv, float z) {
    vec2 cell = 2.0 * P(0).zw;
    float invScale = 1.0 / max(0.03 * z, 1e-4);
    vec4 sum = vec4(0.0);
    float weight = 0.0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 at = uv + vec2(x, y) * cell;
            float w = bilateralWeight(depthAt(at) - z, invScale) *
                    (x == 0 && y == 0 ? 2.0 : 1.0);
            sum += textureLod(image, at, 0.0) * w;
            weight += w;
        }
    }
    return sum / max(weight, 1e-4);
}

// The quarter-size bounced light at full size, the same way.
vec3 bouncedAt(vec2 uv, float z) {
    vec2 cell = 4.0 * P(0).zw;
    float invScale = 1.0 / max(0.05 * z, 1e-4);
    vec3 sum = textureLod(bounced, uv, 0.0).rgb;
    float weight = 1.0;
    for (int i = 0; i < 4; i++) {
        vec2 o = vec2(i == 0 || i == 1 ? -1.0 : 1.0, i == 0 || i == 2 ? -1.0 : 1.0) * cell;
        float w = bilateralWeight(depthAt(uv + o) - z, invScale);
        sum += textureLod(bounced, uv + o, 0.0).rgb * w;
        weight += w;
    }
    return sum / weight;
}

float blurOf(float z, float focusAt) {
    float spread = abs(1.0 - focusAt / max(z, 1e-4));
    return clamp(spread / (0.05 + 0.95 * P(4).w), 0.0, 1.0) * P(5).x;
}

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 lumaTap(vec2 uv) {
    return textureLod(source, uv, 0.0).rgb;
}

// Edges smoothed along their own direction: the stronger luma gradient picks the axis, and the
// pixel takes in its neighbour across the edge as far as the edge's own contrast and, at the best
// quality, its length say.
vec3 smoothEdges(vec2 uv, vec3 c, float z) {
    vec2 tx = P(0).zw;
    float lc = luma(c);
    float ln = luma(lumaTap(uv + vec2(0.0, tx.y)));
    float ls = luma(lumaTap(uv - vec2(0.0, tx.y)));
    float le = luma(lumaTap(uv + vec2(tx.x, 0.0)));
    float lw = luma(lumaTap(uv - vec2(tx.x, 0.0)));
    float lo = min(lc, min(min(ln, ls), min(le, lw)));
    float hi = max(lc, max(max(ln, ls), max(le, lw)));
    float range = hi - lo;
    if (range < max(P(5).w, hi * 0.125) && edgeAt(uv, z) < 0.5) return c;
    float lne = luma(lumaTap(uv + tx));
    float lsw = luma(lumaTap(uv - tx));
    float lse = luma(lumaTap(uv + vec2(tx.x, -tx.y)));
    float lnw = luma(lumaTap(uv + vec2(-tx.x, tx.y)));
    float across = abs(ln + ls - 2.0 * lc) * 2.0 + abs(lne + lse - 2.0 * le) + abs(lnw + lsw - 2.0 * lw);
    float along = abs(le + lw - 2.0 * lc) * 2.0 + abs(lne + lnw - 2.0 * ln) + abs(lse + lsw - 2.0 * ls);
    bool horizontal = across >= along;
    vec2 normal = horizontal ? vec2(0.0, tx.y) : vec2(tx.x, 0.0);
    float a = horizontal ? ln : le;
    float b = horizontal ? ls : lw;
    if (abs(b - lc) > abs(a - lc)) normal = -normal;
    float average = (ln + ls + le + lw) * 0.25;
    float sub = smoothstep(0.0, 1.0, clamp(abs(average - lc) / max(range, 1e-4), 0.0, 1.0));
    float amount = sub * sub * 0.75;
    if (P(6).x > 0.5) {
        // Walk the edge both ways to its ends; a pixel near an end leans on its neighbour more.
        vec2 dir = horizontal ? vec2(tx.x, 0.0) : vec2(0.0, tx.y);
        float edge = (lc + (abs(b - lc) > abs(a - lc) ? b : a)) * 0.5;
        float target = range * 0.25;
        float dn = 8.0, dp = 8.0;
        for (int i = 1; i <= 8; i++) {
            if (dn > 7.5 && abs(luma(lumaTap(uv + 0.5 * normal - dir * float(i))) - edge) > target) dn = float(i);
            if (dp > 7.5 && abs(luma(lumaTap(uv + 0.5 * normal + dir * float(i))) - edge) > target) dp = float(i);
        }
        amount = max(amount, 0.5 - min(dn, dp) / (dn + dp));
    } else {
        amount = max(amount, 0.25);
    }
    vec3 smoothed = textureLod(source, uv + normal * amount, 0.0).rgb;
    return mix(c, smoothed, P(5).z);
}

vec3 rotateHue(vec3 c, float angle) {
    float k = 0.57735027;
    float cs = cos(angle);
    vec3 axis = vec3(k);
    return c * cs + cross(axis, c) * sin(angle) + axis * dot(axis, c) * (1.0 - cs);
}

vec3 filmic(vec3 x) {
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
}

vec3 agx(vec3 x) {
    const mat3 inset = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                            0.0784335999999992, 0.878468636469772, 0.0784336,
                            0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 outset = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                             -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                             -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
    const float lo = -12.47393;
    const float hi = 4.026069;
    vec3 v = (clamp(log2(max(inset * x, vec3(1e-10))), lo, hi) - lo) / (hi - lo);
    vec3 v2 = v * v;
    vec3 v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    return pow(max(outset * v, vec3(0.0)), vec3(2.2));
}

vec3 neutral(vec3 c) {
    const float start = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(c.r, min(c.g, c.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    c -= offset;
    float peak = max(c.r, max(c.g, c.b));
    if (peak < start) return c;
    float d = 1.0 - start;
    float top = 1.0 - d * d / (peak + d - start);
    c *= top / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - top) + 1.0);
    return mix(c, vec3(top), g);
}

// The game's tonemapped frame taken back toward scene light, then through the chosen curve with
// the gain that keeps mid grey (kelir/rust/src/look.rs).
vec3 tonemap(vec3 c) {
    vec3 x = pow(max(c, vec3(0.0)), vec3(2.2));
    float expand = P(1).z;
    x = x / max(1.0 - expand * x, vec3(1.0 - expand));
    x = 0.18 * pow(max(x / 0.18, vec3(0.0)), vec3(1.0 + 0.5 * P(1).w)) * P(1).y;
    vec3 y = P(1).x > 1.5 ? neutral(x) : P(1).x > 0.5 ? agx(x) : filmic(x);
    return pow(clamp(y, 0.0, 1.0), vec3(1.0 / 2.2));
}

// How much of the sky shows here: the far plane, softened over the pixel's neighbours.
float skyAt(vec2 uv) {
    vec2 tx = P(0).zw;
    float n = 0.0;
    for (int i = 0; i < 4; i++) {
        vec2 o = vec2(i == 0 ? 1.0 : i == 1 ? -1.0 : 0.0, i == 2 ? 1.0 : i == 3 ? -1.0 : 0.0) * tx * 2.0;
        n += depthAt(uv + o) >= kSky ? 1.0 : 0.0;
    }
    return (depthAt(uv) >= kSky ? 0.5 : 0.0) + n * 0.125;
}

vec3 look(vec2 uv, vec3 c, float z) {
    if (P(5).z > 0.0) c = smoothEdges(uv, c, z);
    if (P(5).y > 0.0) {
        float sky = skyAt(uv);
        if (sky > 0.0) {
            vec3 g = c * exp2(P(2).x);
            float y = luma(g);
            g = vec3(y) + (g - vec3(y)) * (1.0 + P(2).y);
            g = rotateHue(g, P(2).z);
            g *= mix(vec3(1.0), P(2).w > 0.0 ? vec3(1.08, 1.0, 0.86) : vec3(0.9, 0.98, 1.1), abs(P(2).w));
            c = mix(c, g, sky * P(5).y);
        }
    }
    if (P(3).w > 0.0) {
        vec3 sun = texelFetch(focus, ivec2(0), 0).xyz;
        if (sun.z > 0.0) {
            // Light that sky pixels between here and the sun scatter toward the camera.
            vec2 step = (sun.xy - uv) * P(3).x / 24.0;
            vec2 at = uv;
            float decay = 1.0;
            float sum = 0.0;
            for (int i = 0; i < 24; i++) {
                at += step;
                float open = depthAt(at) >= kSky ? 1.0 : 0.0;
                sum += max(luma(textureLod(source, at, 0.0).rgb) - P(3).y, 0.0) * open * decay;
                decay *= 0.95;
            }
            vec3 tint = mix(vec3(1.0), vec3(1.0, 0.82, 0.55), P(3).z);
            c += tint * (sum / 24.0) * sun.z * P(3).w * 2.0;
        }
    }
    if (P(5).x > 0.0) c = mix(c, tonemap(c), P(5).x);
    if (P(4).w > 0.0) {
        // Lines where depth jumps, and where luma does as far as Detail asks.
        vec2 tx = P(0).zw * P(4).x;
        float lz = depthAt(uv - vec2(tx.x, 0.0));
        float rz = depthAt(uv + vec2(tx.x, 0.0));
        float bz = depthAt(uv - vec2(0.0, tx.y));
        float tz = depthAt(uv + vec2(0.0, tx.y));
        // On the near side only, and only where one neighbour on an axis falls away: a pixel with
        // far depth on both sides is a dither hole (screen-door fades), not a silhouette.
        vec4 far = vec4(lz, rz, bz, tz) - z;
        float jumpX = max(far.x, far.y) - max(min(far.x, far.y), 0.0);
        float jumpY = max(far.z, far.w) - max(min(far.z, far.w), 0.0);
        float jump = max(jumpX, jumpY) / max(z, 1e-4);
        vec3 cl = lumaTap(uv - vec2(tx.x, 0.0));
        vec3 cr = lumaTap(uv + vec2(tx.x, 0.0));
        vec3 cb = lumaTap(uv - vec2(0.0, tx.y));
        vec3 ct = lumaTap(uv + vec2(0.0, tx.y));
        // A depth edge the picture does not show sits behind fog, cloud or glass: no line there.
        vec3 dh = abs(cl - cr);
        vec3 dv = abs(cb - ct);
        float shown = smoothstep(0.03, 0.1, max(max(max(dh.r, dh.g), dh.b), max(max(dv.r, dv.g), dv.b)));
        float depthLine = smoothstep(0.03, 0.12, jump) * shown;
        float gl = abs(luma(cl) - luma(cr));
        float gv = abs(luma(cb) - luma(ct));
        float lumaLine = smoothstep(0.15, 0.4, max(gl, gv)) * P(4).z;
        c *= 1.0 - P(4).y * max(depthLine, lumaLine) * P(4).w;
    }
    return c;
}

void main() {
    vec2 uv = gl_FragCoord.xy * P(0).zw;
    vec3 c = textureLod(source, uv, 0.0).rgb;
    float z = depthAt(uv);
    if (P(7).x > 1.5) {
        c = look(uv, c, z);
    } else if (P(7).x < 0.5) {
        if (P(1).y > 0.0) c *= mix(1.0, occlusionAt(uv, z), P(1).y);
        if (P(7).y > 0.5 && z < kSky) {
            // Shadow and occlusion darken the light, bounce adds onto the albedo guess below,
            // and the reflection lies over the result as it would on a coat of varnish.
            vec2 seen = tracedAt(rtVisibility, uv, z).xy;
            vec4 mirrored = tracedAt(rtReflection, uv, z);
            vec3 light = pow(c, vec3(2.2));
            float y = dot(light, vec3(0.2126, 0.7152, 0.0722));
            vec3 albedo = clamp(light / max(y, 0.02) * 0.4, 0.0, 1.0);
            light = light * seen.x * seen.y + albedo * tracedAt(rtBounce, uv, z).rgb;
            c = pow(light, vec3(1.0 / 2.2));
            c = c * (1.0 - mirrored.a) + mirrored.rgb;
        }
        if (P(6).z > 0.0 && z < kSky) {
            // Onto a guess at the surface's albedo: its own hue at a middling brightness.
            vec3 light = pow(c, vec3(2.2));
            float y = dot(light, vec3(0.2126, 0.7152, 0.0722));
            vec3 albedo = clamp(light / max(y, 0.02) * 0.4, 0.0, 1.0);
            light += albedo * bouncedAt(uv, z) * (P(6).y * P(6).z);
            c = pow(light, vec3(1.0 / 2.2));
        }
        if (P(2).z > 0.0) {
            float f = 1.0 - exp(-P(2).y * max(z - P(2).x, 0.0));
            // The sky keeps some of itself.
            if (z >= kSky) f *= 0.5;
            c = mix(c, P(3).rgb, f * P(2).z);
        }
    } else {
        // A gather by circle of confusion: a tap counts where its own blur reaches this pixel;
        // behind the pixel, only as far as the pixel's blur, so a sharp subject keeps its edges.
        float focusAt = P(4).x > 0.5 ? P(4).y : texelFetch(focus, ivec2(0), 0).r;
        float most = P(4).z * P(0).y;
        float own = blurOf(z, focusAt) * most;
        vec3 sum = c;
        float weight = 1.0;
        float turn = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy,
                                                               vec2(0.06711056, 0.00583715))));
        for (int i = 0; i < kTaps; i++) {
            float a = turn + float(i) * 2.3999632;
            vec2 o = vec2(cos(a), sin(a)) * sqrt((float(i) + 0.5) / float(kTaps)) * most;
            vec2 at = uv + o * P(0).zw;
            float zt = depthAt(at);
            float theirs = blurOf(zt, focusAt) * most;
            float reach = zt < z ? theirs : min(theirs, own);
            float w = clamp(reach - length(o) + 1.0, 0.0, 1.0);
            sum += textureLod(source, at, 0.0).rgb * w;
            weight += w;
        }
        c = sum / weight;
    }
    color = vec4(c, edgeAt(uv, z));
}
