// One level of Kelir's pyramid. The first is a quarter of the frame, each texel the mean of a 4x4
// block (four bilinear taps on pixel corners), stored encoded; the rest halve the level above
// with one tap. Kept in fp16 so blurred gradients do not band.

KELIR_SAMPLER(0, 0) highp sampler2D kelirLevelSource;
layout(location = 0) out vec4 kelirLevel;

PARAMS;  // P(0): xy texel size of the source, z 1 for the first level, w 1 to encode its reads

void main() {
    vec2 p = floor(gl_FragCoord.xy);
    vec3 c;
    if (P(0).z > 0.5) {
        vec2 base = p * 4.0;
        c = 0.25 * (texture(kelirLevelSource, (base + vec2(1.0, 1.0)) * P(0).xy).rgb +
                    texture(kelirLevelSource, (base + vec2(3.0, 1.0)) * P(0).xy).rgb +
                    texture(kelirLevelSource, (base + vec2(1.0, 3.0)) * P(0).xy).rgb +
                    texture(kelirLevelSource, (base + vec2(3.0, 3.0)) * P(0).xy).rgb);
        if (P(0).w > 0.5) c = kelirEncode(c);
    } else {
        c = texture(kelirLevelSource, (p * 2.0 + 1.0) * P(0).xy).rgb;
    }
    kelirLevel = vec4(c, 1.0);
}
