// Global illumination in screen space, at a quarter of the scene's size: light the visible surfaces bounce
// onto each other. Rays leave each pixel's surface in a cosine-weighted direction and march
// through the game's depth; one that passes behind a surface takes that surface's colour as the
// light coming from it. The noise of a few rays per pixel averages out over frames: the result
// before is carried along the game's own motion, wherever the surface it was on is still there
// (alpha keeps each pixel's view distance to tell).
layout(BINDING(0, 0)) uniform sampler2D lin;       // view distance, scene size
layout(BINDING(1, 1)) uniform sampler2D source;    // the scene, display encoded
layout(BINDING(2, 2)) uniform sampler2D history;   // the result before, quarter size
layout(BINDING(3, 3)) uniform sampler2D velocity;  // the game's, packed; or none
layout(location = 0) out vec4 gi;

PARAMS;  // P(0): scene size, 1 / scene size   P(1): x reach (view units), y and z focal scales,
         // w frame   P(2): x 1 when history holds the frame before, y 1 with velocity, z the
         // velocity's zero

const float kSky = 50000.0;
const int kRays = 2;
const int kSteps = 6;
const float kCell = 4.0;  // scene pixels per pixel; scene.frag reads it back the same way

// The position of the texel nearest `uv`, at that texel's centre: a point placed half a texel off
// the depth it read leaves a slanted floor in bands of false occlusion.
vec3 viewAt(vec2 uv) {
    vec2 at = (floor(uv * P(0).xy) + 0.5) * P(0).zw;
    float z = textureLod(lin, at, 0.0).r;
    vec2 ndc = at * 2.0 - 1.0;
    return vec3(ndc * z / P(1).yz, z);
}

vec2 uvOf(vec3 v) {
    return (v.xy * P(1).yz / v.z) * 0.5 + 0.5;
}

float hash(vec2 p, float k) {
    return fract(52.9829189 * fract(dot(p + k * vec2(5.588238, 3.235791),
                                        vec2(0.06711056, 0.00583715))));
}

vec3 linear(vec3 c) {
    return c * c * (0.25 + 0.75 * c);  // close to sRGB's curve, cheap
}

void main() {
    vec2 uv = (floor(gl_FragCoord.xy) * kCell + 0.5) * P(0).zw;
    vec3 p = viewAt(uv);
    if (p.z >= kSky) {
        gi = vec4(0.0, 0.0, 0.0, p.z);
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
    vec3 tangent = normalize(abs(n.y) < 0.99 ? cross(n, vec3(0.0, 1.0, 0.0))
                                             : cross(n, vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(n, tangent);

    float reach = P(1).x;
    vec3 light = vec3(0.0);
    for (int k = 0; k < kRays; k++) {
        float u1 = hash(gl_FragCoord.xy, P(1).w * 3.0 + float(k));
        float u2 = hash(gl_FragCoord.yx + 17.0, P(1).w * 7.0 + float(k));
        float a = 6.2831853 * u2;
        float s = sqrt(u1);
        vec3 d = tangent * (cos(a) * s) + bitangent * (sin(a) * s) + n * sqrt(1.0 - u1);
        float jitter = hash(gl_FragCoord.xy + 31.0, P(1).w + float(k));
        for (int i = 0; i < kSteps; i++) {
            float f = (float(i) + jitter) / float(kSteps);
            float dist = reach * f * f + 0.02 * p.z;
            vec3 q = p + d * dist;
            if (q.z <= 0.0) break;
            vec2 at = uvOf(q);
            if (any(lessThan(at, vec2(0.0))) || any(greaterThan(at, vec2(1.0)))) break;
            float z = textureLod(lin, at, 0.0).r;
            // Behind the surface there, but not so far behind that it is a different one.
            if (z < q.z - 0.02 * q.z && z > q.z - 0.5 * dist - 0.3) {
                light += linear(textureLod(source, at, 0.0).rgb);
                break;
            }
        }
    }
    light /= float(kRays);

    float keep = 0.0;
    vec3 before = vec3(0.0);
    if (P(2).x > 0.5) {
        vec2 from = uv;
        if (P(2).y > 0.5) {
            vec2 v = (textureLod(velocity, uv, 0.0).xy - P(2).z) * 2.0;
            from -= sign(v) * v * v;
        }
        vec4 h = textureLod(history, from, 0.0);
        bool inside = all(greaterThanEqual(from, vec2(0.0))) && all(lessThanEqual(from, vec2(1.0)));
        if (inside && abs(h.a - p.z) < 0.05 * p.z) {
            keep = 0.85;
            before = h.rgb;
        }
    }
    gi = vec4(mix(light, before, keep), p.z);
}
