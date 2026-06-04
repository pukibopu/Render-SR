#include "GBufferPass.hpp"
#include "Mesh.hpp"
#include "Uniforms.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <stdexcept>
#include <string>

namespace rs {

namespace {

constexpr MTL::PixelFormat kRGBFormat    = MTL::PixelFormatRGBA8Unorm_sRGB;
constexpr MTL::PixelFormat kDepthFormat  = MTL::PixelFormatR32Float;       // linear depth output
constexpr MTL::PixelFormat kNormalFormat = MTL::PixelFormatRGBA16Float;
constexpr MTL::PixelFormat kZFormat      = MTL::PixelFormatDepth32Float;   // depth-test

NS::String* nsStr(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

MTL::VertexDescriptor* makeMeshVertexDescriptor() {
    MTL::VertexDescriptor* vd = MTL::VertexDescriptor::alloc()->init();
    vd->attributes()->object(0)->setFormat(MTL::VertexFormatFloat3);
    vd->attributes()->object(0)->setOffset(offsetof(MeshVertex, position));
    vd->attributes()->object(0)->setBufferIndex(0);

    vd->attributes()->object(1)->setFormat(MTL::VertexFormatFloat3);
    vd->attributes()->object(1)->setOffset(offsetof(MeshVertex, normal));
    vd->attributes()->object(1)->setBufferIndex(0);

    vd->attributes()->object(2)->setFormat(MTL::VertexFormatFloat2);
    vd->attributes()->object(2)->setOffset(offsetof(MeshVertex, uv));
    vd->attributes()->object(2)->setBufferIndex(0);

    vd->layouts()->object(0)->setStride(sizeof(MeshVertex));
    vd->layouts()->object(0)->setStepFunction(MTL::VertexStepFunctionPerVertex);
    return vd;
}

MTL::Texture* makeRenderTarget(MTL::Device* dev,
                               MTL::PixelFormat fmt,
                               uint32_t w, uint32_t h,
                               bool sampleable)
{
    MTL::TextureDescriptor* td = MTL::TextureDescriptor::texture2DDescriptor(fmt, w, h, false);
    NS::UInteger usage = MTL::TextureUsageRenderTarget;
    if (sampleable) usage |= MTL::TextureUsageShaderRead;
    td->setUsage(usage);
    td->setStorageMode(MTL::StorageModePrivate);
    return dev->newTexture(td);
}

void throwIfError(MTL::RenderPipelineState* pso, NS::Error* err, const char* what) {
    if (pso) return;
    std::string msg = err && err->localizedDescription()
        ? err->localizedDescription()->utf8String() : "unknown error";
    throw std::runtime_error(std::string("failed to build ") + what + ": " + msg);
}

}

GBufferPass::GBufferPass(MTL::Device* device,
                         MTL::Library* library,
                         int swapchainPixelFormat,
                         uint32_t lowW,  uint32_t lowH,
                         uint32_t highW, uint32_t highH)
    : m_device(device), m_library(library),
      m_lowW(lowW), m_lowH(lowH), m_highW(highW), m_highH(highH)
{
    buildDepthState();
    buildSampler();
    buildScenePipelines();
    buildBlitPipelines(swapchainPixelFormat);
    buildLowResTextures();
    buildHighResTextures();
}

GBufferPass::~GBufferPass() {
    if (m_lowRGB)    { m_lowRGB->release();    m_lowRGB    = nullptr; }
    if (m_lowDepth)  { m_lowDepth->release();  m_lowDepth  = nullptr; }
    if (m_lowNormal) { m_lowNormal->release(); m_lowNormal = nullptr; }
    if (m_lowZ)      { m_lowZ->release();      m_lowZ      = nullptr; }
    if (m_highRGB)   { m_highRGB->release();   m_highRGB   = nullptr; }
    if (m_highZ)     { m_highZ->release();     m_highZ     = nullptr; }

    if (m_blitRGB)     { m_blitRGB->release();     m_blitRGB     = nullptr; }
    if (m_blitDepth)   { m_blitDepth->release();   m_blitDepth   = nullptr; }
    if (m_blitNormal)  { m_blitNormal->release();  m_blitNormal  = nullptr; }
    if (m_mrtPipeline) { m_mrtPipeline->release(); m_mrtPipeline = nullptr; }
    if (m_rgbPipeline) { m_rgbPipeline->release(); m_rgbPipeline = nullptr; }

    if (m_sampler)    { m_sampler->release();    m_sampler    = nullptr; }
    if (m_depthState) { m_depthState->release(); m_depthState = nullptr; }
}

void GBufferPass::buildDepthState() {
    MTL::DepthStencilDescriptor* d = MTL::DepthStencilDescriptor::alloc()->init();
    d->setDepthCompareFunction(MTL::CompareFunctionLess);
    d->setDepthWriteEnabled(true);
    m_depthState = m_device->newDepthStencilState(d);
    d->release();
    if (!m_depthState) throw std::runtime_error("failed to build depth-stencil state");
}

void GBufferPass::buildSampler() {
    MTL::SamplerDescriptor* sd = MTL::SamplerDescriptor::alloc()->init();
    sd->setMinFilter(MTL::SamplerMinMagFilterLinear);
    sd->setMagFilter(MTL::SamplerMinMagFilterLinear);
    sd->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    sd->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    m_sampler = m_device->newSamplerState(sd);
    sd->release();
    if (!m_sampler) throw std::runtime_error("failed to build sampler state");
}

void GBufferPass::buildScenePipelines() {
    MTL::Function* vfn       = m_library->newFunction(nsStr("gbuffer_vertex"));
    MTL::Function* fgbuffer  = m_library->newFunction(nsStr("gbuffer_fragment"));
    MTL::Function* frgb      = m_library->newFunction(nsStr("rgb_fragment"));
    if (!vfn || !fgbuffer || !frgb) {
        if (vfn)      vfn->release();
        if (fgbuffer) fgbuffer->release();
        if (frgb)     frgb->release();
        throw std::runtime_error("gbuffer shader functions not found in metallib");
    }

    MTL::VertexDescriptor* vd = makeMeshVertexDescriptor();

    // MRT pipeline: writes RGB + linear depth + view-space normal.
    {
        MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vfn);
        desc->setFragmentFunction(fgbuffer);
        desc->setVertexDescriptor(vd);
        desc->colorAttachments()->object(0)->setPixelFormat(kRGBFormat);
        desc->colorAttachments()->object(1)->setPixelFormat(kDepthFormat);
        desc->colorAttachments()->object(2)->setPixelFormat(kNormalFormat);
        desc->setDepthAttachmentPixelFormat(kZFormat);

        NS::Error* err = nullptr;
        m_mrtPipeline = m_device->newRenderPipelineState(desc, &err);
        desc->release();
        throwIfError(m_mrtPipeline, err, "MRT pipeline state");
    }

    // RGB-only pipeline: high-res pass produces no auxiliary buffers.
    {
        MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vfn);
        desc->setFragmentFunction(frgb);
        desc->setVertexDescriptor(vd);
        desc->colorAttachments()->object(0)->setPixelFormat(kRGBFormat);
        desc->setDepthAttachmentPixelFormat(kZFormat);

        NS::Error* err = nullptr;
        m_rgbPipeline = m_device->newRenderPipelineState(desc, &err);
        desc->release();
        throwIfError(m_rgbPipeline, err, "RGB-only pipeline state");
    }

    vd->release();
    vfn->release();
    fgbuffer->release();
    frgb->release();
}

void GBufferPass::buildBlitPipelines(int swapchainPixelFormat) {
    const auto swapFmt = static_cast<MTL::PixelFormat>(swapchainPixelFormat);

    MTL::Function* vfn  = m_library->newFunction(nsStr("blit_vertex"));
    MTL::Function* frgb = m_library->newFunction(nsStr("blit_rgb_fragment"));
    MTL::Function* fdep = m_library->newFunction(nsStr("blit_depth_fragment"));
    MTL::Function* fnor = m_library->newFunction(nsStr("blit_normal_fragment"));
    if (!vfn || !frgb || !fdep || !fnor) {
        if (vfn)  vfn->release();
        if (frgb) frgb->release();
        if (fdep) fdep->release();
        if (fnor) fnor->release();
        throw std::runtime_error("blit shader functions not found in metallib");
    }

    auto buildOne = [&](MTL::Function* frag, const char* tag) {
        MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
        desc->setVertexFunction(vfn);
        desc->setFragmentFunction(frag);
        desc->colorAttachments()->object(0)->setPixelFormat(swapFmt);
        NS::Error* err = nullptr;
        MTL::RenderPipelineState* p = m_device->newRenderPipelineState(desc, &err);
        desc->release();
        throwIfError(p, err, tag);
        return p;
    };

    m_blitRGB    = buildOne(frgb, "blit-rgb pipeline state");
    m_blitDepth  = buildOne(fdep, "blit-depth pipeline state");
    m_blitNormal = buildOne(fnor, "blit-normal pipeline state");

    vfn->release();
    frgb->release();
    fdep->release();
    fnor->release();
}

void GBufferPass::buildLowResTextures() {
    m_lowRGB    = makeRenderTarget(m_device, kRGBFormat,    m_lowW, m_lowH, true);
    m_lowDepth  = makeRenderTarget(m_device, kDepthFormat,  m_lowW, m_lowH, true);
    m_lowNormal = makeRenderTarget(m_device, kNormalFormat, m_lowW, m_lowH, true);
    m_lowZ      = makeRenderTarget(m_device, kZFormat,      m_lowW, m_lowH, false);
    if (!m_lowRGB || !m_lowDepth || !m_lowNormal || !m_lowZ) {
        throw std::runtime_error("failed to allocate low-res G-buffer textures");
    }
}

void GBufferPass::buildHighResTextures() {
    m_highRGB = makeRenderTarget(m_device, kRGBFormat, m_highW, m_highH, true);
    m_highZ   = makeRenderTarget(m_device, kZFormat,   m_highW, m_highH, false);
    if (!m_highRGB || !m_highZ) {
        throw std::runtime_error("failed to allocate high-res textures");
    }
}

void GBufferPass::encodeLowRes(MTL::CommandBuffer* cmd,
                               const Mesh& mesh, const Uniforms& u) const
{
    MTL::RenderPassDescriptor* desc = MTL::RenderPassDescriptor::alloc()->init();

    auto* c0 = desc->colorAttachments()->object(0);
    c0->setTexture(m_lowRGB);
    c0->setLoadAction(MTL::LoadActionClear);
    c0->setStoreAction(MTL::StoreActionStore);
    c0->setClearColor(MTL::ClearColor::Make(0.10, 0.15, 0.20, 1.0));

    auto* c1 = desc->colorAttachments()->object(1);
    c1->setTexture(m_lowDepth);
    c1->setLoadAction(MTL::LoadActionClear);
    c1->setStoreAction(MTL::StoreActionStore);
    // 0 = sky / background; geometry writes positive eye-space distances.
    c1->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0));

    auto* c2 = desc->colorAttachments()->object(2);
    c2->setTexture(m_lowNormal);
    c2->setLoadAction(MTL::LoadActionClear);
    c2->setStoreAction(MTL::StoreActionStore);
    c2->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 0.0));

    auto* z = desc->depthAttachment();
    z->setTexture(m_lowZ);
    z->setLoadAction(MTL::LoadActionClear);
    z->setStoreAction(MTL::StoreActionDontCare);
    z->setClearDepth(1.0);

    MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(desc);
    enc->setRenderPipelineState(m_mrtPipeline);
    enc->setDepthStencilState(m_depthState);
    enc->setCullMode(MTL::CullModeBack);
    enc->setFrontFacingWinding(MTL::WindingCounterClockwise);

    enc->setVertexBuffer(mesh.vertexBuffer(), 0, 0);
    enc->setVertexBytes(&u, sizeof(u), 1);
    enc->setFragmentBytes(&u, sizeof(u), 1);

    enc->drawIndexedPrimitives(
        MTL::PrimitiveTypeTriangle,
        mesh.indexCount(),
        MTL::IndexTypeUInt32,
        mesh.indexBuffer(),
        NS::UInteger(0));
    enc->endEncoding();
    desc->release();
}

void GBufferPass::encodeHighRes(MTL::CommandBuffer* cmd,
                                const Mesh& mesh, const Uniforms& u) const
{
    MTL::RenderPassDescriptor* desc = MTL::RenderPassDescriptor::alloc()->init();

    auto* c0 = desc->colorAttachments()->object(0);
    c0->setTexture(m_highRGB);
    c0->setLoadAction(MTL::LoadActionClear);
    c0->setStoreAction(MTL::StoreActionStore);
    c0->setClearColor(MTL::ClearColor::Make(0.10, 0.15, 0.20, 1.0));

    auto* z = desc->depthAttachment();
    z->setTexture(m_highZ);
    z->setLoadAction(MTL::LoadActionClear);
    z->setStoreAction(MTL::StoreActionDontCare);
    z->setClearDepth(1.0);

    MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(desc);
    enc->setRenderPipelineState(m_rgbPipeline);
    enc->setDepthStencilState(m_depthState);
    enc->setCullMode(MTL::CullModeBack);
    enc->setFrontFacingWinding(MTL::WindingCounterClockwise);

    enc->setVertexBuffer(mesh.vertexBuffer(), 0, 0);
    enc->setVertexBytes(&u, sizeof(u), 1);
    enc->setFragmentBytes(&u, sizeof(u), 1);

    enc->drawIndexedPrimitives(
        MTL::PrimitiveTypeTriangle,
        mesh.indexCount(),
        MTL::IndexTypeUInt32,
        mesh.indexBuffer(),
        NS::UInteger(0));
    enc->endEncoding();
    desc->release();
}

void GBufferPass::encodeBlitToSwapchain(MTL::CommandBuffer* cmd,
                                        MTL::Texture* dst,
                                        DebugBlit which,
                                        float depthMin, float depthMax) const
{
    MTL::Texture* src = nullptr;
    MTL::RenderPipelineState* pso = nullptr;
    switch (which) {
        case DebugBlit::RGB:    src = m_lowRGB;    pso = m_blitRGB;    break;
        case DebugBlit::Depth:  src = m_lowDepth;  pso = m_blitDepth;  break;
        case DebugBlit::Normal: src = m_lowNormal; pso = m_blitNormal; break;
    }

    MTL::RenderPassDescriptor* desc = MTL::RenderPassDescriptor::alloc()->init();
    auto* c0 = desc->colorAttachments()->object(0);
    c0->setTexture(dst);
    c0->setLoadAction(MTL::LoadActionClear);
    c0->setStoreAction(MTL::StoreActionStore);
    c0->setClearColor(MTL::ClearColor::Make(0.0, 0.0, 0.0, 1.0));

    MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(desc);
    enc->setRenderPipelineState(pso);
    enc->setFragmentTexture(src, 0);
    enc->setFragmentSamplerState(m_sampler, 0);
    if (which == DebugBlit::Depth) {
        float range[2] = { depthMin, depthMax };
        enc->setFragmentBytes(range, sizeof(range), 0);
    }
    enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
    enc->endEncoding();
    desc->release();
}

}
