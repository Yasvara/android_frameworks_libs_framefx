// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
//
// The game's velocity buffer, raw, at the centre of each finest motion block: xy as the image
// lies, zw with its rows flipped. Which of the two lines up with the scene, and how the game
// encodes motion in it, is fitted on the CPU against the block search (luwes/vulkan.cpp).
layout(BINDING(0, 0)) uniform sampler2D velocity;
layout(location = 0) out vec4 raw;

PARAMS;  // P(0): xy 1 / grid size

void main() {
    vec2 uv = gl_FragCoord.xy * P(0).xy;
    raw = vec4(textureLod(velocity, uv, 0.0).xy, textureLod(velocity, vec2(uv.x, 1.0 - uv.y), 0.0).xy);
}
