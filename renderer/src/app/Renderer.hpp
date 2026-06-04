#pragma once

#include "Camera.hpp"
#include "GBufferPass.hpp"

#include <memory>

struct GLFWwindow;

namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
}
namespace CA { class MetalLayer; }

namespace rs {

class Mesh;

class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame();

    Camera&       camera()       { return m_camera; }
    const Camera& camera() const { return m_camera; }

    void setDebugBlit(GBufferPass::DebugBlit b) { m_debugBlit = b; }

    const char* deviceName() const;

private:
    void loadLibrary();

    GLFWwindow*        m_window  = nullptr;
    MTL::Device*       m_device  = nullptr;
    MTL::CommandQueue* m_queue   = nullptr;
    CA::MetalLayer*    m_layer   = nullptr;
    MTL::Library*      m_library = nullptr;

    std::unique_ptr<GBufferPass> m_gbuffer;
    std::unique_ptr<Mesh>        m_mesh;
    Camera                       m_camera;

    GBufferPass::DebugBlit m_debugBlit = GBufferPass::DebugBlit::RGB;
};

}
