#pragma once

struct GLFWwindow;

namespace MTL { class Device; class CommandQueue; }
namespace CA  { class MetalLayer; }

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
    GLFWwindow*        m_window = nullptr;
    MTL::Device*       m_device = nullptr;
    MTL::CommandQueue* m_queue  = nullptr;
    CA::MetalLayer*    m_layer  = nullptr;
};

}
