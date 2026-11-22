// The game's depth as view distance, from its projection: clip depth d' = c + b / z, so
// z = b / (d' - c). The far plane and beyond read as kFar.
layout(BINDING(0, 0)) uniform sampler2D depth;
layout(location = 0) out vec4 view;

PARAMS;  // P(0): output size, c, b   P(1): x 1 when the game stores GL clip depth, y 1 when the
         // depth lies across the output: read with x and y swapped

const float kFar = 60000.0;

void main() {
    vec2 at = P(1).y > 0.5 ? gl_FragCoord.yx / P(0).yx : gl_FragCoord.xy / P(0).xy;
    float d = textureLod(depth, at, 0.0).r;
    if (P(1).x > 0.5) d = 2.0 * d - 1.0;
    float den = d - P(0).z;
    float z = abs(den) < 1e-7 ? kFar : P(0).w / den;
    view = vec4(z > 0.0 ? min(z, kFar) : kFar);
}
