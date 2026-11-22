// The game's scene with x and y swapped: a game that draws upright and turns its frame for a
// pre-rotated swapchain, read in the swapchain's axes. Which way it turned is left to the flips
// the output passes vote for.
layout(BINDING(0, 0)) uniform sampler2D scene;
layout(location = 0) out vec4 color;

PARAMS;  // P(0): xy this image's size

void main() {
    color = texture(scene, gl_FragCoord.yx / P(0).yx);
}
