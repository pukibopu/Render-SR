#include "Renderer.hpp"
#include "CocoaBridge.h"

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

namespace rs {

namespace {

constexpr MTL::PixelFormat kSwapchainFormat = MTL::PixelFormatBGRA8Unorm_sRGB;

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

struct Vertex {
    float pos[2];
    float color[3];
};

}

Renderer::Renderer(GLFWwindow* window) : m_window(window) {
    m_device = MTL::CreateSystemDefaultDevice();
    if (!m_device) {
        throw std::runtime_error("MTL::CreateSystemDefaultDevice returned null");
    }

    m_queue = m_device->newCommandQueue();
    if (!m_queue) {
        m_device->release();
        m_device = nullptr;
        throw std::runtime_error("device->newCommandQueue returned null");
    }

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

    loadLibrary();
    buildPipeline();
    buildVertexBuffer();
}

Renderer::~Renderer() {
    if (m_vbuf)     { m_vbuf->release();     m_vbuf     = nullptr; }
    if (m_pipeline) { m_pipeline->release(); m_pipeline = nullptr; }
    if (m_library)  { m_library->release();  m_library  = nullptr; }
    if (m_queue)    { m_queue->release();    m_queue    = nullptr; }
    if (m_device)   { m_device->release();   m_device   = nullptr; }
}

const char* Renderer::deviceName() const {
    return m_device ? m_device->name()->utf8String() : "<null>";
}

void Renderer::loadLibrary() {
    std::string path = executableDir() + "/default.metallib";

    NS::String* nsPath = nsStr(path.c_str());
    NS::URL* url = NS::URL::fileURLWithPath(nsPath);

    NS::Error* err = nullptr;
    m_library = m_device->newLibrary(url, &err);
    if (!m_library) {
        std::string msg = err && err->localizedDescription()
            ? err->localizedDescription()->utf8String()
            : "unknown error";
        throw std::runtime_error("failed to load metallib at " + path + ": " + msg);
    }
}

void Renderer::buildPipeline() {
    MTL::Function* vfn = m_library->newFunction(nsStr("triangle_vertex"));
    MTL::Function* ffn = m_library->newFunction(nsStr("triangle_fragment"));
    if (!vfn || !ffn) {
        if (vfn) vfn->release();
        if (ffn) ffn->release();
        throw std::runtime_error("triangle_vertex / triangle_fragment not found in metallib");
    }

    MTL::RenderPipelineDescriptor* desc = MTL::RenderPipelineDescriptor::alloc()->init();
    desc->setVertexFunction(vfn);
    desc->setFragmentFunction(ffn);
    desc->colorAttachments()->object(0)->setPixelFormat(kSwapchainFormat);

    NS::Error* err = nullptr;
    m_pipeline = m_device->newRenderPipelineState(desc, &err);

    desc->release();
    vfn->release();
    ffn->release();

    if (!m_pipeline) {
        std::string msg = err && err->localizedDescription()
            ? err->localizedDescription()->utf8String()
            : "unknown error";
        throw std::runtime_error("failed to build render pipeline state: " + msg);
    }
}

void Renderer::buildVertexBuffer() {
    const Vertex verts[3] = {
        {{ 0.0f,  0.6f}, {1.0f, 0.0f, 0.0f}},
        {{-0.6f, -0.5f}, {0.0f, 1.0f, 0.0f}},
        {{ 0.6f, -0.5f}, {0.0f, 0.0f, 1.0f}},
    };
    m_vbuf = m_device->newBuffer(verts, sizeof(verts), MTL::ResourceStorageModeShared);
    if (!m_vbuf) {
        throw std::runtime_error("failed to allocate vertex buffer");
    }
}

void Renderer::drawFrame() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    if (w > 0 && h > 0) {
        CGSize current = m_layer->drawableSize();
        if (current.width != (CGFloat)w || current.height != (CGFloat)h) {
            m_layer->setDrawableSize(CGSize{(CGFloat)w, (CGFloat)h});
        }
    }

    CA::MetalDrawable* drawable = m_layer->nextDrawable();
    if (drawable) {
        MTL::CommandBuffer* cmd = m_queue->commandBuffer();

        MTL::RenderPassDescriptor* desc = MTL::RenderPassDescriptor::alloc()->init();
        auto* color = desc->colorAttachments()->object(0);
        color->setTexture(drawable->texture());
        color->setLoadAction(MTL::LoadActionClear);
        color->setStoreAction(MTL::StoreActionStore);
        color->setClearColor(MTL::ClearColor::Make(0.10, 0.15, 0.20, 1.0));

        MTL::RenderCommandEncoder* enc = cmd->renderCommandEncoder(desc);
        enc->setRenderPipelineState(m_pipeline);
        enc->setVertexBuffer(m_vbuf, 0, 0);
        enc->drawPrimitives(MTL::PrimitiveTypeTriangle, NS::UInteger(0), NS::UInteger(3));
        enc->endEncoding();

        cmd->presentDrawable(drawable);
        cmd->commit();

        desc->release();
    }

    pool->release();
}

}
