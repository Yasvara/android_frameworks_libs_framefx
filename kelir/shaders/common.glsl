// Kelir: what every snippet shares. Values are the target's encoded RGB unless named linear.
//
// FRAME, per present:
//   0: buffer size px, display size px (display orientation)
//   1: x quarter turns from buffer to display, y 1 when buffer row 0 is the bottom (GL),
//      z split position in display x (negative: off), w present seed (an exact integer)
//   2: xyz luma weights of the target's primaries, w half an LSB of the target's depth
//   3: x 1 when the source's alpha marks the game's depth edges
// PLAN, per stack (kelir/plan.h PackPlan): 2 vec4s of parameters per detail slot (0..5) and per
//   screen slot (6..13); intensities of the detail slots (14.xyz) and screen slots (15).

KELIR_PLAN;
KELIR_FRAME;

const int KELIR_SHARPEN = 7;
const int KELIR_DETAILS = 8;
const int KELIR_TILT_SHIFT = 9;
const int KELIR_PAINTERLY = 10;
const int KELIR_WATERCOLOR = 11;
const int KELIR_VIGNETTE = 12;
const int KELIR_FILM_GRAIN = 13;
const int KELIR_LETTERBOX = 14;
const int KELIR_OLD_FILM = 15;
const int KELIR_RETRO = 16;
const int KELIR_HALFTONE = 17;

const int KELIR_ROLE_FULL = 0;
const int KELIR_ROLE_STAGE_A = 1;
const int KELIR_ROLE_STAGE_B = 2;

vec4 kelirDetailParams(int slot, int part) { return PLAN(slot * 2 + part); }
vec4 kelirScreenParams(int slot, int part) { return PLAN(6 + slot * 2 + part); }
float kelirDetailIntensity(int slot) { return PLAN(14)[slot]; }
float kelirScreenIntensity(int slot) { return PLAN(15)[slot]; }

vec2 kelirBufferSize() { return FRAME(0).xy; }
vec2 kelirDisplaySize() { return FRAME(0).zw; }
uint kelirSeed() { return uint(FRAME(1).w); }

float kelirLuma(vec3 c) { return dot(c, FRAME(2).xyz); }

vec3 kelirDecode(vec3 e) {
    vec3 a = abs(e);
    vec3 l = mix(pow((a + 0.055) / 1.055, vec3(2.4)), a / 12.92, lessThanEqual(a, vec3(0.04045)));
    return sign(e) * l;
}

vec3 kelirEncode(vec3 l) {
    vec3 a = abs(l);
    vec3 e = mix(1.055 * pow(a, vec3(1.0 / 2.4)) - 0.055, a * 12.92, lessThanEqual(a, vec3(0.0031308)));
    return sign(l) * e;
}

// Buffer pixel -> display uv, origin at the display's top left as the user holds it.
vec2 kelirDisplayUv(vec2 px) {
    vec2 b = px / FRAME(0).xy;
    if (FRAME(1).y > 0.5) b.y = 1.0 - b.y;
    int turns = int(FRAME(1).x);
    if (turns == 1) return vec2(b.y, 1.0 - b.x);
    if (turns == 2) return 1.0 - b;
    if (turns == 3) return vec2(1.0 - b.y, b.x);
    return b;
}

// Display uv -> buffer texture coordinates.
vec2 kelirBufferUv(vec2 d) {
    int turns = int(FRAME(1).x);
    vec2 b = d;
    if (turns == 1) b = vec2(1.0 - d.y, d.x);
    if (turns == 2) b = 1.0 - d;
    if (turns == 3) b = vec2(d.y, 1.0 - d.x);
    if (FRAME(1).y > 0.5) b.y = 1.0 - b.y;
    return b;
}

// PCG3D (Jarzynski and Olano): integer in, three independent uniform values out.
uvec3 kelirPcg(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

vec3 kelirRandom(uvec3 v) {
    return vec3(kelirPcg(v) >> 8u) * (1.0 / 16777216.0);
}

vec3 kelirRandomAt(vec2 cell, uint salt) {
    return kelirRandom(uvec3(uvec2(ivec2(cell) + 32768), salt));
}
