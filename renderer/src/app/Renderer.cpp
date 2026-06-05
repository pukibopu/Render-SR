#include "Renderer.hpp"
#include "CocoaBridge.h"
#include "Mesh.hpp"
#include "Uniforms.h"
#include "io/FrameWriter.hpp"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <GLFW/glfw3.h>

#include <mach-o/dyld.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace rs {

namespace {

constexpr MTL::PixelFormat kSwapchainFormat = MTL::PixelFormatBGRA8Unorm_sRGB;

constexpr uint32_t kLowResW  = 960;
constexpr uint32_t kLowResH  = 540;
constexpr uint32_t kHighResW = 1920;
constexpr uint32_t kHighResH = 1080;

constexpr float kRenderAspect = float(kLowResW) / float(kLowResH);

std::string executableDir() {
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buf(size, '\0');
    if (_NSGetExecutablePath(buf.data(), &size) != 0) {
        throw std::runtime_error("_NSGetExecutablePath failed");
    }
    buf.resize(std::strlen(buf.c_str()));

    char resolved[PATH_MAX];
    const char* abs = realpath(buf.c_str(), resolved) ? resolved : buf.c_str();
    std::string s(abs);
    auto pos = s.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : s.substr(0, pos);
}

NS::String* nsStr(const char* s) {
    return NS::String::string(s, NS::UTF8StringEncoding);
}

simd_float3x3 makeNormalMatrix(simd_float4x4 mv) {
    simd_float4x4 t = simd_transpose(simd_inverse(mv));
    simd_float3x3 r;
    r.columns[0] = simd_make_float3(t.columns[0].x, t.columns[0].y, t.columns[0].z);
    r.columns[1] = simd_make_float3(t.columns[1].x, t.columns[1].y, t.columns[1].z);
    r.columns[2] = simd_make_float3(t.columns[2].x, t.columns[2].y, t.columns[2].z);
    return r;
}

// Apple Silicon native half: this file is arm64 macOS only (enforced in CMake).
inline float halfToFloat(std::uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return static_cast<float>(h);
}

}

Renderer::Renderer(GLFWwindow* window) : m_window(window) {
    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) throw std::runtime_error("MTL::CreateSystemDefaultDevice returned null");

    m_queue = m_device->newCommandQueue();
    if (!m_queue) {
        m_device->release(); m_device = nullptr;
        throw std::runtime_error("device->newCommandQueue returned null");
    }

    if (m_window) {
        void* opaque = cocoa_bridge_attach_metal_layer(window);
        if (!opaque) {
            m_queue->release();  m_queue  = nullptr;
            m_device->release(); m_device = nullptr;
            throw std::runtime_error("failed to attach CAMetalLayer to NSWindow");
        }
        m_layer = reinterpret_cast<CA::MetalLayer*>(opaque);

        m_layer->setDevice(m_device);
        m_layer->setPixelFormat(kSwapchainFormat);
        m_layer->setFramebufferOnly(true);

        int w = 0, h = 0;
        glfwGetFramebufferSize(window, &w, &h);
        m_layer->setDrawableSize(CGSize{(CGFloat)w, (CGFloat)h});
    }

    m_camera.setAspect(kRenderAspect);

    loadLibrary();

    m_gbuffer = std::make_unique<GBufferPass>(
        m_device, m_library, static_cast<int>(kSwapchainFormat),
        kLowResW, kLowResH, kHighResW, kHighResH);

    m_scene = std::make_unique<Scene>(m_device);
}

Renderer::~Renderer() {
    m_scene.reset();
    m_gbuffer.reset();
    if (m_library) { m_library->release(); m_library = nullptr; }
    if (m_queue)   { m_queue->release();   m_queue   = nullptr; }
    if (m_device)  { m_device->release();  m_device  = nullptr; }
}

const char* Renderer::deviceName() const {
    return m_device ? m_device->name()->utf8String() : "<null>";
}

void Renderer::loadLibrary() {
    const std::string path = executableDir() + "/default.metallib";
    NS::URL* url = NS::URL::fileURLWithPath(nsStr(path.c_str()));
    NS::Error* err = nullptr;
    m_library = m_device->newLibrary(url, &err);
    if (!m_library) {
        std::string msg = err && err->localizedDescription()
            ? err->localizedDescription()->utf8String() : "unknown error";
        throw std::runtime_error("failed to load metallib at " + path + ": " + msg);
    }
}

void Renderer::buildDrawItems(std::vector<DrawItem>& out) const {
    const simd_float4x4 view = m_camera.view();
    const simd_float4x4 proj = m_camera.proj();

    // Light direction is world-space in the scene; transform into view space
    // once (rotation only, view has no scale) so shading is stable as the
    // camera orbits.
    const simd_float3x3 viewRot = makeNormalMatrix(view);
    const simd_float3 lightDirView =
        simd_normalize(simd_mul(viewRot, m_scene->lightDirWorld()));
    const float ambient = m_scene->ambient();

    const auto& objects = m_scene->objects();
    out.clear();
    out.reserve(objects.size());
    for (const SceneObject& o : objects) {
        Uniforms u;
        u.model = o.model;
        u.view  = view;
        u.proj  = proj;
        u.normalMatrix = makeNormalMatrix(simd_mul(view, o.model));
        u.lightDirView = lightDirView;
        u.ambient = ambient;
        u.albedo = o.material.albedo;
        u.specularStrength = o.material.specularStrength;
        out.push_back(DrawItem{o.mesh, u});
    }
}

void Renderer::drawFrame() {
    if (!m_window) return;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    if (w <= 0 || h <= 0) { pool->release(); return; }

    CGSize current = m_layer->drawableSize();
    if (current.width != (CGFloat)w || current.height != (CGFloat)h) {
        m_layer->setDrawableSize(CGSize{(CGFloat)w, (CGFloat)h});
    }

    CA::MetalDrawable* drawable = m_layer->nextDrawable();
    if (!drawable) { pool->release(); return; }

    std::vector<DrawItem> items; buildDrawItems(items);

    MTL::CommandBuffer* cmd = m_queue->commandBuffer();
    m_gbuffer->encodeLowRes (cmd, items);
    m_gbuffer->encodeHighRes(cmd, items);

    // Scene spans a wider depth range than the old single torus; widen the
    // grayscale debug remap so the floor and far cube stay visible.
    constexpr float kDepthVizMin = 1.5f;
    constexpr float kDepthVizMax = 14.0f;
    m_gbuffer->encodeBlitToSwapchain(
        cmd, drawable->texture(), m_debugBlit, kDepthVizMin, kDepthVizMax);

    cmd->presentDrawable(drawable);
    cmd->commit();

    pool->release();
}

io::CameraSnapshot Renderer::dumpFrame(const std::filesystem::path& outDir,
                                       int pathId, int frameInPath) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    std::vector<DrawItem> items; buildDrawItems(items);

    MTL::CommandBuffer* cmd = m_queue->commandBuffer();
    m_gbuffer->encodeLowRes (cmd, items);
    m_gbuffer->encodeHighRes(cmd, items);
    m_gbuffer->encodeReadback(cmd);
    cmd->commit();
    cmd->waitUntilCompleted();

    const std::uint32_t lowW  = m_gbuffer->lowResWidth();
    const std::uint32_t lowH  = m_gbuffer->lowResHeight();
    const std::uint32_t highW = m_gbuffer->highResWidth();
    const std::uint32_t highH = m_gbuffer->highResHeight();

    // Convert RGBA16Float normal staging into a tightly packed (H, W, 3)
    // float32 view-space buffer, preserving the [-1, 1] component range
    // (no remapping into colour space).
    const std::uint16_t* halfN = m_gbuffer->lowNormalHalfHost();
    const std::size_t px = std::size_t(lowW) * std::size_t(lowH);
    std::vector<float> normalRGB(px * 3);
    for (std::size_t i = 0; i < px; ++i) {
        normalRGB[3*i + 0] = halfToFloat(halfN[4*i + 0]);
        normalRGB[3*i + 1] = halfToFloat(halfN[4*i + 1]);
        normalRGB[3*i + 2] = halfToFloat(halfN[4*i + 2]);
    }

    io::CameraSnapshot snap{
        .fov_y      = m_camera.fovY(),
        .aspect     = m_camera.aspect(),
        .near_plane = m_camera.nearPlane(),
        .far_plane  = m_camera.farPlane(),
        .target     = m_camera.target(),
        .azimuth    = m_camera.azimuth(),
        .elevation  = m_camera.elevation(),
        .distance   = m_camera.distance(),
        .eye        = m_camera.eye(),
        .view       = m_camera.view(),
    };

    io::writeFrame(outDir, pathId, frameInPath,
                   lowW, lowH, highW, highH,
                   m_gbuffer->lowRGBHost(),
                   m_gbuffer->lowDepthHost(),
                   normalRGB.data(),
                   m_gbuffer->highRGBHost(),
                   snap);

    pool->release();
    return snap;
}

std::uint32_t Renderer::lowResWidth () const { return m_gbuffer->lowResWidth();  }
std::uint32_t Renderer::lowResHeight() const { return m_gbuffer->lowResHeight(); }
std::uint32_t Renderer::highResWidth () const { return m_gbuffer->highResWidth();  }
std::uint32_t Renderer::highResHeight() const { return m_gbuffer->highResHeight(); }

}
