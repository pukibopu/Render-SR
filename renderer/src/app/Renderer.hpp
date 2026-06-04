#pragma once

struct GLFWwindow;

namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class RenderPipelineState;
    class Buffer;
}
namespace CA { class MetalLayer; }

namespace rs {

class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame();

    const char* deviceName() const;

private:
    void loadLibrary();
    void buildPipeline();
    void buildVertexBuffer();

    GLFWwindow*                m_window   = nullptr;
    MTL::Device*               m_device   = nullptr;
    MTL::CommandQueue*         m_queue    = nullptr;
    CA::MetalLayer*            m_layer    = nullptr;
    MTL::Library*              m_library  = nullptr;
    MTL::RenderPipelineState*  m_pipeline = nullptr;
    MTL::Buffer*               m_vbuf     = nullptr;
};

}
