/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include "src/texture.h"

#include <GLES2/gl2.h>

#include <algorithm>

namespace framefx::texture {

bool Override(const TextureSettings& s, const Limits& l, VkSamplerCreateInfo* info) {
    if (!s.on() || info->pNext != nullptr || info->flags != 0 || info->unnormalizedCoordinates ||
        info->compareEnable || info->magFilter != VK_FILTER_LINEAR ||
        info->minFilter != VK_FILTER_LINEAR || info->maxLod <= 0.0f) {
        return false;
    }
    const bool af = s.anisotropy > 1 && l.anisotropy;
    if (!af && s.lodBiasEighths == 0) return false;
    if (af) {
        const float asked = info->anisotropyEnable ? info->maxAnisotropy : 1.0f;
        info->anisotropyEnable = VK_TRUE;
        info->maxAnisotropy = std::max(asked, std::min(static_cast<float>(s.anisotropy),
                                                       l.maxAnisotropy));
        info->mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
    if (s.lodBiasEighths != 0) {
        info->mipLodBias = std::clamp(info->mipLodBias + s.lodBiasEighths / 8.0f, -l.maxLodBias,
                                      l.maxLodBias);
    }
    return true;
}

bool OverrideGl(const TextureSettings& s, float maxAnisotropy, int minFilter, GlFilter* out) {
    if (!s.on() || (minFilter != GL_LINEAR_MIPMAP_LINEAR && minFilter != GL_LINEAR_MIPMAP_NEAREST)) {
        return false;
    }
    const bool af = s.anisotropy > 1 && maxAnisotropy > 1.0f;
    if (!af && s.lodBiasEighths == 0) return false;
    out->minFilter = af ? GL_LINEAR_MIPMAP_LINEAR : minFilter;
    out->anisotropy = af ? std::min(static_cast<float>(s.anisotropy), maxAnisotropy) : 0.0f;
    // QCOM_texture_lod_bias takes the same range as GLES's own sampler LOD.
    out->lodBias = std::clamp(s.lodBiasEighths / 8.0f, -4.0f, 4.0f);
    return true;
}

}  // namespace framefx::texture
