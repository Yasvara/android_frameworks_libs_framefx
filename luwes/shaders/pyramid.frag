// One level of the luma pyramid: each texel averages a 2x2 tile of the level above, which one
// bilinear tap at the tile's centre does. The first level reads the captured frame and converts
// to luma; the target is aligned up to whole motion blocks, and the sampler clamps the padding.
layout(BINDING(0, 0)) uniform sampler2D source;
layout(location = 0) out vec4 luma;

PARAMS;  // P(0): xy texel size of `source`, z 1 to convert colour to luma

void main() {
    vec4 c = texture(source, (floor(gl_FragCoord.xy) * 2.0 + 1.0) * P(0).xy);
    float y = P(0).z > 0.5 ? dot(c.rgb, vec3(0.299, 0.587, 0.114)) : c.r;
    luma = vec4(y, 0.0, 0.0, 1.0);
}
