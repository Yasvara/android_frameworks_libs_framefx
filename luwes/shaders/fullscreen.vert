// One triangle covering the target; no vertex input.
void main() {
    vec2 p = vec2(VERTEX_ID == 1 ? 3.0 : -1.0, VERTEX_ID == 2 ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
