// Ambient occlusion from the game's depth at half the scene's size: how much of the hemisphere
// over each surface the geometry within the radius hides. Positions come from view distance and
// the projection's focal scales; the normal from the depth beside the pixel, the smaller step on
// each axis so silhouettes do not bend it. Eight taps on a spiral turned per pixel; scene.frag
// smooths the noise as it upsamples.
layout(BINDING(0, 0)) uniform sampler2D lin;
layout(location = 0) out vec4 ao;

PARAMS;  // P(0): scene size, 1 / scene size   P(1): x radius (view units), y and z focal scales

const int kTaps = 8;
const float kSky = 50000.0;

// The position of the texel nearest `uv`, at that texel's centre: a point placed half a texel off
// the depth it read leaves a slanted floor in bands of false occlusion.
vec3 viewAt(vec2 uv) {
    vec2 at = (floor(uv * P(0).xy) + 0.5) * P(0).zw;
    float z = textureLod(lin, at, 0.0).r;
    vec2 ndc = at * 2.0 - 1.0;
    return vec3(ndc * z / P(1).yz, z);
}

void main() {
    vec2 uv = (floor(gl_FragCoord.xy) * 2.0 + 0.5) * P(0).zw;
    vec3 p = viewAt(uv);
    if (p.z >= kSky) {
        ao = vec4(1.0);
        return;
    }
    vec2 tx = P(0).zw;
    vec3 l = viewAt(uv - vec2(tx.x, 0.0));
    vec3 r = viewAt(uv + vec2(tx.x, 0.0));
    vec3 b = viewAt(uv - vec2(0.0, tx.y));
    vec3 t = viewAt(uv + vec2(0.0, tx.y));
    vec3 dx = abs(r.z - p.z) < abs(p.z - l.z) ? r - p : p - l;
    vec3 dy = abs(t.z - p.z) < abs(p.z - b.z) ? t - p : p - b;
    vec3 n = normalize(cross(dx, dy));
    if (dot(n, p) > 0.0) n = -n;
    float radius = P(1).x;
    // The radius in uv at this distance: clip units per view unit are focal / z, uv half that.
    vec2 reach = min(0.5 * radius * abs(P(1).yz) / p.z, vec2(0.1));
    float turn = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy,
                                                           vec2(0.06711056, 0.00583715))));
    float occlusion = 0.0;
    for (int i = 0; i < kTaps; i++) {
        float a = turn + float(i) * 2.3999632;
        vec2 o = vec2(cos(a), sin(a)) * sqrt((float(i) + 0.5) / float(kTaps)) * reach;
        vec3 v = viewAt(uv + o) - p;
        float dist2 = dot(v, v);
        float falloff = max(0.0, 1.0 - dist2 / (radius * radius));
        occlusion += max(0.0, dot(n, v) * inversesqrt(max(dist2, 1e-8)) - 0.1) * falloff;
    }
    ao = vec4(clamp(1.0 - 1.5 * occlusion / float(kTaps), 0.0, 1.0));
}
