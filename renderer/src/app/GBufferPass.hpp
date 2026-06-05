#pragma once

#include "Uniforms.h"

#include <cstdint>
#include <vector>

namespace MTL {
    class Device;
    class Library;
    class CommandBuffer;
    class RenderPipelineState;
    class DepthStencilState;
    class SamplerState;
    class Texture;
    class Buffer;
}

namespace rs {

class Mesh;

// One drawable: a mesh plus the per-object uniforms (model + material + the
// shared view/proj/light) to render it with. All items in a pass are drawn
// into the same attachments in one render encoder (no clear between objects).
struct DrawItem {
    const Mesh* mesh;
    Uniforms    uniforms;
};

// Off-screen multi-render-target scene pass plus a fullscreen blit that
// copies one of the low-res attachments onto the swapchain for visual
// inspection.
//
//   Low-res pass  : color(0) RGBA8Unorm_sRGB  shaded RGB
//                   color(1) R32Float          linear eye-space depth
//                   color(2) RGBA16Float       view-space normal
//                   depth    Depth32Float      (test only, not stored)
//   High-res pass : color(0) RGBA8Unorm_sRGB  shaded RGB
//                   depth    Depth32Float      (test only, not stored)
//
// Sizes are fixed at construction; the fragment formats are intentional
// (see shaders/README.md).
class GBufferPass {
public:
    enum class DebugBlit { RGB, Depth, Normal };

    GBufferPass(MTL::Device* device,
                MTL::Library* library,
                int swapchainPixelFormat,
                uint32_t lowW,  uint32_t lowH,
                uint32_t highW, uint32_t highH);

    ~GBufferPass();

    GBufferPass(const GBufferPass&) = delete;
    GBufferPass& operator=(const GBufferPass&) = delete;

    void encodeLowRes (MTL::CommandBuffer* cmd, const std::vector<DrawItem>& items) const;
    void encodeHighRes(MTL::CommandBuffer* cmd, const std::vector<DrawItem>& items) const;

    // Blit one low-res attachment onto `dst` (typically the swapchain texture).
    // depthMin/depthMax map the R32Float depth range to [0,1] for visualization;
    // ignored for RGB and Normal.
    void encodeBlitToSwapchain(MTL::CommandBuffer* cmd,
                               MTL::Texture* dst,
                               DebugBlit which,
                               float depthMin, float depthMax) const;

    // Copy all four output attachments (low-res RGB/depth/normal + high-res
    // RGB) from private textures into host-visible staging buffers. The
    // staging buffers are valid to read after the command buffer that this
    // call was encoded onto has completed (waitUntilCompleted on caller).
    void encodeReadback(MTL::CommandBuffer* cmd) const;

    // Host pointers into the staging buffers. Layouts are tightly packed,
    // row-major, no row padding:
    //   lowRGBHost   : uint8_t[lowH * lowW * 4]   (RGBA8, sRGB-encoded)
    //   lowDepthHost : float  [lowH * lowW]
    //   lowNormalHalfHost : uint16_t[lowH * lowW * 4] (RGBA16Float)
    //   highRGBHost  : uint8_t[highH * highW * 4] (RGBA8, sRGB-encoded)
    const std::uint8_t*  lowRGBHost()        const;
    const float*         lowDepthHost()      const;
    const std::uint16_t* lowNormalHalfHost() const;
    const std::uint8_t*  highRGBHost()       const;

    uint32_t lowResWidth () const { return m_lowW; }
    uint32_t lowResHeight() const { return m_lowH; }
    uint32_t highResWidth () const { return m_highW; }
    uint32_t highResHeight() const { return m_highH; }

private:
    void buildScenePipelines();
    void buildBlitPipelines(int swapchainPixelFormat);
    void buildDepthState();
    void buildSampler();
    void buildLowResTextures();
    void buildHighResTextures();
    void buildReadbackBuffers();

    MTL::Device*  m_device  = nullptr;
    MTL::Library* m_library = nullptr;

    // Scene pipelines (forward + MRT; vertex shader is shared)
    MTL::RenderPipelineState* m_mrtPipeline = nullptr;  // 3 color attachments
    MTL::RenderPipelineState* m_rgbPipeline = nullptr;  // 1 color attachment

    // Blit pipelines (fullscreen triangle, no vertex buffer)
    MTL::RenderPipelineState* m_blitRGB    = nullptr;
    MTL::RenderPipelineState* m_blitDepth  = nullptr;
    MTL::RenderPipelineState* m_blitNormal = nullptr;

    MTL::DepthStencilState* m_depthState = nullptr;
    MTL::SamplerState*      m_sampler    = nullptr;

    // Low-res attachments
    MTL::Texture* m_lowRGB    = nullptr;
    MTL::Texture* m_lowDepth  = nullptr;   // linear, R32Float (output)
    MTL::Texture* m_lowNormal = nullptr;
    MTL::Texture* m_lowZ      = nullptr;   // Depth32Float, test-only

    // High-res attachments
    MTL::Texture* m_highRGB = nullptr;
    MTL::Texture* m_highZ   = nullptr;     // Depth32Float, test-only

    // Host-visible staging buffers for readback.
    MTL::Buffer* m_lowRGBBuf    = nullptr;
    MTL::Buffer* m_lowDepthBuf  = nullptr;
    MTL::Buffer* m_lowNormalBuf = nullptr;
    MTL::Buffer* m_highRGBBuf   = nullptr;

    uint32_t m_lowW = 0,  m_lowH = 0;
    uint32_t m_highW = 0, m_highH = 0;
};

}
