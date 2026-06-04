#include "Renderer.hpp"
#include "CocoaBridge.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <GLFW/glfw3.h>

#include <stdexcept>

namespace rs {

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
    m_layer->setPixelFormat(MTL::PixelFormatBGRA8Unorm_sRGB);
    m_layer->setFramebufferOnly(true);

    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    m_layer->setDrawableSize(CGSize{(CGFloat)w, (CGFloat)h});
}

Renderer::~Renderer() {
    if (m_queue)  { m_queue->release();  m_queue  = nullptr; }
    if (m_device) { m_device->release(); m_device = nullptr; }
}

const char* Renderer::deviceName() const {
    return m_device ? m_device->name()->utf8String() : "<null>";
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
        enc->endEncoding();

        cmd->presentDrawable(drawable);
        cmd->commit();

        desc->release();
    }

    pool->release();
}

}
