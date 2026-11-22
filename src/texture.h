/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#pragma once

#include <vulkan/vulkan.h>

#include "src/channel.h"

// Texture filtering the game didn't ask for: anisotropic and trilinear, with a sharper LOD.
namespace framefx::texture {

struct Limits {
    bool anisotropy = false;  // the device was made with samplerAnisotropy
    float maxAnisotropy = 1.0f;
    float maxLodBias = 0.0f;
};

// The game's sampler as `settings` ask, for linear mipmapped samplers only: never pixel art, UI
// without mips, shadow compares, lookups by texel or anything chained. False, and `info` left
// alone, for any other.
bool Override(const TextureSettings& settings, const Limits& limits, VkSamplerCreateInfo* info);

// GLES: what goes with a minifying filter a game sets on a texture or sampler.
struct GlFilter {
    int minFilter;        // the filter to set in place of the game's
    float anisotropy;     // GL_TEXTURE_MAX_ANISOTROPY_EXT to set; 0: none
    float lodBias;        // GL_TEXTURE_LOD_BIAS_QCOM to set (textures only); 0: none
};
// For linear mipmapped filters only, as on Vulkan. `maxAnisotropy` is the driver's limit, 0
// without the extension. False, and `out` unset, for any other filter.
bool OverrideGl(const TextureSettings& settings, float maxAnisotropy, int minFilter,
                GlFilter* out);

}  // namespace framefx::texture
