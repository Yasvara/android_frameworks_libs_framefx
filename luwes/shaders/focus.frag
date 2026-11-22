// Auto focus: the view distance in the middle of the scene (the median of five taps), eased in
// over a few frames so the focus does not jump when something crosses the centre.
layout(BINDING(0, 0)) uniform sampler2D lin;
layout(BINDING(1, 1)) uniform sampler2D before;
layout(location = 0) out vec4 focus;

void main() {
    float z[5];
    z[0] = textureLod(lin, vec2(0.5, 0.5), 0.0).r;
    z[1] = textureLod(lin, vec2(0.46, 0.5), 0.0).r;
    z[2] = textureLod(lin, vec2(0.54, 0.5), 0.0).r;
    z[3] = textureLod(lin, vec2(0.5, 0.45), 0.0).r;
    z[4] = textureLod(lin, vec2(0.5, 0.55), 0.0).r;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4 - i; j++) {
            float a = min(z[j], z[j + 1]);
            z[j + 1] = max(z[j], z[j + 1]);
            z[j] = a;
        }
    }
    float prev = texelFetch(before, ivec2(0), 0).r;
    focus = vec4(prev > 0.0 ? mix(prev, z[2], 0.12) : z[2]);
}
