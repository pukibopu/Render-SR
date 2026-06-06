#pragma once

#include "Camera.hpp"
#include "GBufferPass.hpp"
#include "Scene.hpp"
#include "io/FrameWriter.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

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

    // Render one frame at the camera's current pose and write its outputs under
    // `outDir`, named frame_{pathId:02d}_{frameInPath:04d}. Synchronous: blocks
    // until the GPU has finished. Returns the camera snapshot for the manifest.
    io::CameraSnapshot dumpFrame(const std::filesystem::path& outDir,
                                 int pathId, int frameInPath);

    Camera&       camera()       { return m_camera; }
    const Camera& camera() const { return m_camera; }

    // Select a deterministic scene variant (0 = fixed base scene). Subsequent
    // frames render this variant's object layout + lighting. Used to give each
    // dataset-v2 camera path its own scene without rebuilding GPU resources.
    void setSceneVariant(std::uint32_t seed);

    std::uint32_t lowResWidth () const;
    std::uint32_t lowResHeight() const;
    std::uint32_t highResWidth () const;
    std::uint32_t highResHeight() const;

    void setDebugBlit(GBufferPass::DebugBlit b) { m_debugBlit = b; }

    const char* deviceName() const;

private:
    void loadLibrary();
    // Build one DrawItem per scene object at the camera's current pose.
    void buildDrawItems(std::vector<DrawItem>& out) const;

    GLFWwindow*        m_window  = nullptr;
    MTL::Device*       m_device  = nullptr;
    MTL::CommandQueue* m_queue   = nullptr;
    CA::MetalLayer*    m_layer   = nullptr;       // nullptr in headless
    MTL::Library*      m_library = nullptr;

    std::unique_ptr<GBufferPass> m_gbuffer;
    std::unique_ptr<Scene>       m_scene;
    Camera                       m_camera;

    GBufferPass::DebugBlit m_debugBlit = GBufferPass::DebugBlit::RGB;
};

}
