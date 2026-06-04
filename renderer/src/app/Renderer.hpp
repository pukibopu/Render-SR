#pragma once

#include "Camera.hpp"
#include "GBufferPass.hpp"

#include <filesystem>
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

// Pass nullptr for `window` to construct in headless mode (no swapchain;
// drawFrame() is a no-op). dumpFrame() works in either mode.
class Renderer {
public:
    explicit Renderer(GLFWwindow* window);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Render one frame to the offscreen attachments and blit the selected
    // debug attachment to the swapchain. No-op in headless mode.
    void drawFrame();

    // Render one frame and write rgb_low.png, rgb_high.png, depth.npy,
    // normal.npy, meta.json under `outDir` using `frameIndex` as the suffix.
    // Synchronous: blocks until the GPU has finished.
    void dumpFrame(const std::filesystem::path& outDir, int frameIndex);

    Camera&       camera()       { return m_camera; }
    const Camera& camera() const { return m_camera; }

    void setDebugBlit(GBufferPass::DebugBlit b) { m_debugBlit = b; }

    const char* deviceName() const;

private:
    void loadLibrary();
    void buildUniforms(struct Uniforms& u) const;

    GLFWwindow*        m_window  = nullptr;
    MTL::Device*       m_device  = nullptr;
    MTL::CommandQueue* m_queue   = nullptr;
    CA::MetalLayer*    m_layer   = nullptr;       // nullptr in headless
    MTL::Library*      m_library = nullptr;

    std::unique_ptr<GBufferPass> m_gbuffer;
    std::unique_ptr<Mesh>        m_mesh;
    Camera                       m_camera;

    GBufferPass::DebugBlit m_debugBlit = GBufferPass::DebugBlit::RGB;
};

}
