// What the game's HUD puts over its 3D, from the real frame (ungraded) and the game's own stretch
// of its scene before the HUD. The real frame is R = a P + (1 - a) S for a HUD of colour P and
// coverage a, so d = R - S = a (P - S), and the frame with any 3D X under the HUD is
//   a P + (1 - a) X = (1 - a) X + layer,  layer = (1 - a) d + a R  (output.frag).
// Both are stored: layer as layer / 2 + 0.5, mask = a.
//
// The HUD holds still while the scene moves, so between frames d changes by -a times the scene's
// change: a is their least-squares ratio, kept per pixel over the frames before. Where the scene
// has not moved enough to tell, a rises from P(1).x to P(1).y of the largest channel of |d|.
//
// What was learnt predicts this frame: d moves by -a times the scene's change. A pixel further
// off than P(2).w changed its HUD (a dialog opened, the game went to another screen) and starts
// over, or the frames before would show through as a shadow until they decayed.
layout(BINDING(0, 0)) uniform sampler2D real;
layout(BINDING(1, 1)) uniform sampler2D scene;
layout(BINDING(2, 2)) uniform sampler2D history;  // last frame's matte
layout(location = 0) out vec4 layer;
layout(location = 1) out vec4 mask;
// x, y: luma of d and of the scene; z, w: decayed sums of -dd ds and ds ds.
layout(location = 2) out vec4 matte;

PARAMS;  // P(0): frame size, scene x and y flipped (0 or 1)   P(1): x scene floor, y opaque HUD
         // P(2): x history valid, y decay per frame, z scene change worth trusting,
         //       w the most a pixel may differ from what its history predicts
         // P(3): xy the real frame's size when it is not the frame's (0 when it is)

void main() {
    vec2 uv = gl_FragCoord.xy / P(0).xy;
    ivec2 q = ivec2(gl_FragCoord.xy);
    vec3 stretched;
    vec3 r;
    if (P(3).x > 0.5) {
        // The real frame resampled bilinearly to the frame's size, and the game's stretch the
        // same way: the stretch at the real frame's pixel centres, then bilinear between them.
        vec2 size = P(3).xy;
        vec2 p = uv * size - 0.5;
        vec2 i = floor(p);
        vec2 f = p - i;
        vec2 t00 = (i + 0.5) / size;
        vec2 t11 = (i + 1.5) / size;
        vec3 s00 = texture(scene, mix(t00, 1.0 - t00, P(0).zw)).rgb;
        vec3 s10 = texture(scene, mix(vec2(t11.x, t00.y), 1.0 - vec2(t11.x, t00.y), P(0).zw)).rgb;
        vec3 s01 = texture(scene, mix(vec2(t00.x, t11.y), 1.0 - vec2(t00.x, t11.y), P(0).zw)).rgb;
        vec3 s11 = texture(scene, mix(t11, 1.0 - t11, P(0).zw)).rgb;
        stretched = mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
        r = texture(real, uv).rgb;
    } else {
        // Bilinear, as the game's HUD pass reads it.
        stretched = texture(scene, mix(uv, 1.0 - uv, P(0).zw)).rgb;
        r = texelFetch(real, q, 0).rgb;
    }
    vec3 d = r - stretched;
    float m = max(abs(d.r), max(abs(d.g), abs(d.b)));

    const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);
    float dy = dot(d, kLuma);
    float sy = dot(stretched, kLuma);
    vec4 before = P(2).x > 0.5 ? texelFetch(history, q, 0) : vec4(dy, sy, 0.0, 0.0);
    float seen = smoothstep(P(1).x, P(1).y, m);
    float ds = sy - before.y;
    float dd = dy - before.x;
    float held = before.w > 1e-6 ? clamp(before.z / before.w, 0.0, 1.0) : seen;
    bool broke = abs(dd + held * ds) > P(2).w;
    float num = broke ? 0.0 : before.z * P(2).y - dd * ds;
    float den = broke ? 0.0 : before.w * P(2).y + ds * ds;
    matte = vec4(dy, sy, num, den);

    float a = mix(seen, clamp(num / max(den, 1e-6), 0.0, 1.0), den / (den + P(2).z));
    // Below the floor the difference is resampling, not HUD.
    d *= smoothstep(0.25 * P(1).x, P(1).x, m);
    layer = vec4(mix(d, r, a) * 0.5 + 0.5, 1.0);
    mask = vec4(a);
}
