/*
 * SPDX-FileCopyrightText: IRedDragonICY
 * SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0
 */

#include <GLES2/gl2.h>
#include <gtest/gtest.h>

#include <cstring>

#include "src/channel.h"
#include "src/texture.h"

namespace framefx {
namespace {

VkSamplerCreateInfo Trilinear() {
    VkSamplerCreateInfo s = {};
    s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s.maxLod = VK_LOD_CLAMP_NONE;
    return s;
}

constexpr texture::Limits kOnyx = {true, 16.0f, 15.99f};

TEST(TextureSection, Decodes) {
    const uint8_t ok[] = {1, 16, static_cast<uint8_t>(-4), 0};
    TextureSettings t;
    ASSERT_TRUE(DecodeTexture(ok, sizeof(ok), &t));
    EXPECT_EQ(t.anisotropy, 16);
    EXPECT_EQ(t.lodBiasEighths, -4);
    EXPECT_TRUE(t.on());
}

TEST(TextureSection, RejectsAnythingElse) {
    TextureSettings t;
    const uint8_t major[] = {2, 16, 0, 0};
    const uint8_t odd[] = {1, 3, 0, 0};
    const uint8_t blunter[] = {1, 0, 1, 0};
    const uint8_t far[] = {1, 0, static_cast<uint8_t>(-17), 0};
    const uint8_t pad[] = {1, 0, 0, 4};
    const uint8_t adaptiveAtFull[] = {1, 0, 0, 0x80};
    for (const uint8_t* bad : {major, odd, blunter, far, pad, adaptiveAtFull}) {
        EXPECT_FALSE(DecodeTexture(bad, 4, &t));
    }
    EXPECT_FALSE(DecodeTexture(major, 3, &t));
}

TEST(TextureSection, DecodesShading) {
    TextureSettings t;
    const uint8_t fixed[] = {1, 0, 0, 2};
    ASSERT_TRUE(DecodeTexture(fixed, 4, &t));
    EXPECT_EQ(t.shadingRate, 2);
    EXPECT_FALSE(t.shadingAdaptive);
    EXPECT_TRUE(t.shading());
    EXPECT_FALSE(t.on());
    const uint8_t adaptive[] = {1, 16, 0, 0x83};
    ASSERT_TRUE(DecodeTexture(adaptive, 4, &t));
    EXPECT_EQ(t.shadingRate, 3);
    EXPECT_TRUE(t.shadingAdaptive);
    EXPECT_TRUE(t.on());
}

TEST(TextureOverride, MipmappedLinearGetsAnisotropyAndTrilinear) {
    VkSamplerCreateInfo s = Trilinear();
    ASSERT_TRUE(texture::Override({16, -4}, kOnyx, &s));
    EXPECT_TRUE(s.anisotropyEnable);
    EXPECT_EQ(s.maxAnisotropy, 16.0f);
    EXPECT_EQ(s.mipmapMode, VK_SAMPLER_MIPMAP_MODE_LINEAR);
    EXPECT_FLOAT_EQ(s.mipLodBias, -0.5f);
}

TEST(TextureOverride, KeepsTheGamesHigherAnisotropy) {
    VkSamplerCreateInfo s = Trilinear();
    s.anisotropyEnable = VK_TRUE;
    s.maxAnisotropy = 16.0f;
    ASSERT_TRUE(texture::Override({4, 0}, kOnyx, &s));
    EXPECT_EQ(s.maxAnisotropy, 16.0f);
}

TEST(TextureOverride, CappedByTheDevice) {
    VkSamplerCreateInfo s = Trilinear();
    ASSERT_TRUE(texture::Override({16, 0}, {true, 8.0f, 4.0f}, &s));
    EXPECT_EQ(s.maxAnisotropy, 8.0f);
    s = Trilinear();
    s.mipLodBias = -3.9f;
    ASSERT_TRUE(texture::Override({0, -16}, {true, 8.0f, 4.0f}, &s));
    EXPECT_FLOAT_EQ(s.mipLodBias, -4.0f);
}

TEST(TextureOverride, WithoutTheFeatureOnlyTheBias) {
    VkSamplerCreateInfo s = Trilinear();
    EXPECT_FALSE(texture::Override({16, 0}, {false, 16.0f, 15.99f}, &s));
    ASSERT_TRUE(texture::Override({16, -8}, {false, 16.0f, 15.99f}, &s));
    EXPECT_FALSE(s.anisotropyEnable);
    EXPECT_EQ(s.mipmapMode, VK_SAMPLER_MIPMAP_MODE_NEAREST);
    EXPECT_FLOAT_EQ(s.mipLodBias, -1.0f);
}

TEST(TextureOverride, LeavesOtherSamplersAlone) {
    const auto untouched = [](VkSamplerCreateInfo s) {
        const VkSamplerCreateInfo before = s;
        EXPECT_FALSE(texture::Override({16, -8}, kOnyx, &s));
        EXPECT_EQ(memcmp(&before, &s, sizeof(s)), 0);
    };
    VkSamplerCreateInfo pixel = Trilinear();
    pixel.magFilter = VK_FILTER_NEAREST;
    untouched(pixel);
    VkSamplerCreateInfo noMips = Trilinear();
    noMips.maxLod = 0.0f;
    untouched(noMips);
    VkSamplerCreateInfo shadow = Trilinear();
    shadow.compareEnable = VK_TRUE;
    untouched(shadow);
    VkSamplerCreateInfo texel = Trilinear();
    texel.unnormalizedCoordinates = VK_TRUE;
    untouched(texel);
    VkSamplerReductionModeCreateInfo reduction = {};
    reduction.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO;
    VkSamplerCreateInfo chained = Trilinear();
    chained.pNext = &reduction;
    untouched(chained);
    VkSamplerCreateInfo off = Trilinear();
    EXPECT_FALSE(texture::Override({}, kOnyx, &off));
}

TEST(TextureGl, TrilinearAnisotropicWithBias) {
    TextureSettings t;
    t.anisotropy = 16;
    t.lodBiasEighths = -4;
    texture::GlFilter f;
    ASSERT_TRUE(texture::OverrideGl(t, 16.0f, GL_LINEAR_MIPMAP_NEAREST, &f));
    EXPECT_EQ(f.minFilter, GL_LINEAR_MIPMAP_LINEAR);
    EXPECT_EQ(f.anisotropy, 16.0f);
    EXPECT_EQ(f.lodBias, -0.5f);
}

TEST(TextureGl, ClampsToTheDriver) {
    TextureSettings t;
    t.anisotropy = 16;
    texture::GlFilter f;
    ASSERT_TRUE(texture::OverrideGl(t, 8.0f, GL_LINEAR_MIPMAP_LINEAR, &f));
    EXPECT_EQ(f.anisotropy, 8.0f);
    EXPECT_EQ(f.lodBias, 0.0f);
}

TEST(TextureGl, LeavesOtherFiltersAlone) {
    TextureSettings t;
    t.anisotropy = 16;
    t.lodBiasEighths = -4;
    texture::GlFilter f;
    for (int filter : {GL_NEAREST, GL_LINEAR, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR}) {
        EXPECT_FALSE(texture::OverrideGl(t, 16.0f, filter, &f)) << filter;
    }
    EXPECT_FALSE(texture::OverrideGl(TextureSettings(), 16.0f, GL_LINEAR_MIPMAP_LINEAR, &f));
}

TEST(TextureGl, BiasOnlyWithoutTheExtension) {
    TextureSettings t;
    t.anisotropy = 16;
    t.lodBiasEighths = -8;
    texture::GlFilter f;
    ASSERT_TRUE(texture::OverrideGl(t, 0.0f, GL_LINEAR_MIPMAP_NEAREST, &f));
    EXPECT_EQ(f.minFilter, GL_LINEAR_MIPMAP_NEAREST);
    EXPECT_EQ(f.anisotropy, 0.0f);
    EXPECT_EQ(f.lodBias, -1.0f);
}

}  // namespace
}  // namespace framefx
