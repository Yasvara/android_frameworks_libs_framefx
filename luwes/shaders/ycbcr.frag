// A decoder's frame as RGB: the immutable sampler converts from Y'CbCr, so this only picks the texel.
layout(BINDING(0, 0)) uniform sampler2D source;
layout(location = 0) out vec4 colour;

PARAMS;  // P(0): xy texel size of `source`, zw the crop's corner in texels. P(1): xy crop texels per
         // output pixel

void main() {
    colour = vec4(texture(source, (gl_FragCoord.xy * P(1).xy + P(0).zw) * P(0).xy).rgb, 1.0);
}
