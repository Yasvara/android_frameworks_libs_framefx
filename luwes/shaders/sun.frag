// Where the sun stands on the game's scene: the brightness-weighted centre of the brightest sky,
// sampled on a coarse grid and eased over frames. x y its position in the scene's UV, z how
// strongly it shows (0 with no bright sky in view).
layout(BINDING(0, 0)) uniform sampler2D source;
layout(BINDING(1, 1)) uniform sampler2D lin;
layout(BINDING(2, 2)) uniform sampler2D unused0;
layout(BINDING(3, 3)) uniform sampler2D before;  // 1 x 1, the last frame's
layout(BINDING(4, 4)) uniform sampler2D unused1;
layout(location = 0) out vec4 sun;

// P(0): scene size, 1 / size   P(1)..P(6): kelir::PackSceneLook   P(7): unused
PARAMS8;

const float kSky = 50000.0;
const int kColumns = 32;
const int kRows = 18;

void main() {
    float threshold = P(3).y;
    vec3 sum = vec3(0.0);
    float brightest = 0.0;
    for (int y = 0; y < kRows; y++) {
        for (int x = 0; x < kColumns; x++) {
            vec2 uv = (vec2(x, y) + 0.5) / vec2(kColumns, kRows);
            if (textureLod(lin, uv, 0.0).r < kSky) continue;
            float l = dot(textureLod(source, uv, 0.0).rgb, vec3(0.2126, 0.7152, 0.0722));
            float w = pow(max(l - threshold * 0.8, 0.0), 4.0);
            sum += vec3(uv * w, w);
            brightest = max(brightest, l);
        }
    }
    vec4 now = sum.z > 0.0 ? vec4(sum.xy / sum.z, smoothstep(threshold, 1.0, brightest), 1.0)
                           : vec4(0.5, 0.5, 0.0, 1.0);
    vec4 prev = texelFetch(before, ivec2(0), 0);
    // A sun that just appeared starts where it is; one in view glides.
    sun = prev.z > 0.0 && now.z > 0.0 ? vec4(mix(prev.xy, now.xy, 0.2), mix(prev.z, now.z, 0.2), 1.0)
                                      : vec4(now.xy, mix(prev.z, now.z, 0.3), 1.0);
}
