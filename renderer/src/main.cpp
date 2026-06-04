#include "app/Renderer.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>

namespace {

constexpr int kInitialWidth  = 1280;
constexpr int kInitialHeight = 720;

struct Args {
    bool headless = false;
    int  frames   = 1;
    std::filesystem::path out = "output_buffers";
    rs::GBufferPass::DebugBlit debugBlit = rs::GBufferPass::DebugBlit::RGB;
};

void glfwErrorCallback(int code, const char* description) {
    std::fprintf(stderr, "glfw error %d: %s\n", code, description);
}

void pollCameraInput(GLFWwindow* window, rs::Camera& cam, float dt) {
    const float orbitRate = 1.8f;
    const float zoomRate  = 1.6f;

    if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) cam.orbitAzimuth(-orbitRate * dt);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) cam.orbitAzimuth( orbitRate * dt);
    if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) cam.orbitElevation( orbitRate * dt);
    if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) cam.orbitElevation(-orbitRate * dt);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS) {
        cam.scaleDistance(1.0f / (1.0f + zoomRate * dt));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS) {
        cam.scaleDistance(1.0f + zoomRate * dt);
    }
}

std::optional<rs::GBufferPass::DebugBlit> parseDebugBlit(const char* v) {
    if (std::strcmp(v, "rgb")    == 0) return rs::GBufferPass::DebugBlit::RGB;
    if (std::strcmp(v, "depth")  == 0) return rs::GBufferPass::DebugBlit::Depth;
    if (std::strcmp(v, "normal") == 0) return rs::GBufferPass::DebugBlit::Normal;
    return std::nullopt;
}

void printUsage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s [--debug-blit {rgb|depth|normal}]\n"
        "       %s --headless --frames N [--out PATH]\n"
        "\n"
        "  --debug-blit  which low-res G-buffer attachment to display on screen\n"
        "                (defaults to rgb; ignored in headless mode)\n"
        "  --headless    render without a window; write each frame to PATH\n"
        "  --frames N    number of frames to render in headless mode\n"
        "  --out PATH    output directory root (defaults to ./output_buffers)\n",
        argv0, argv0);
}

bool parseArgs(int argc, char** argv, Args& out_args) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: %s requires an argument\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--debug-blit") {
            const char* v = next("--debug-blit");
            if (!v) return false;
            auto parsed = parseDebugBlit(v);
            if (!parsed) {
                std::fprintf(stderr,
                    "error: --debug-blit expects rgb|depth|normal, got '%s'\n", v);
                return false;
            }
            out_args.debugBlit = *parsed;
        } else if (a == "--headless") {
            out_args.headless = true;
        } else if (a == "--frames") {
            const char* v = next("--frames");
            if (!v) return false;
            char* end = nullptr;
            long n = std::strtol(v, &end, 10);
            if (!end || *end != '\0' || n <= 0) {
                std::fprintf(stderr, "error: --frames expects a positive integer, got '%s'\n", v);
                return false;
            }
            out_args.frames = static_cast<int>(n);
        } else if (a == "--out") {
            const char* v = next("--out");
            if (!v) return false;
            out_args.out = std::filesystem::path(v);
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            return false;
        }
    }
    if (!out_args.headless && out_args.frames != 1) {
        std::fprintf(stderr, "error: --frames only applies in --headless mode\n");
        return false;
    }
    return true;
}

int runHeadless(const Args& args) {
    rs::Renderer renderer(nullptr);
    std::printf("Metal device: %s\n", renderer.deviceName());
    std::printf("Writing %d frame%s under: %s\n",
                args.frames, args.frames == 1 ? "" : "s",
                args.out.string().c_str());

    rs::Camera& cam = renderer.camera();
    // Deterministic orbit: full circle across N frames so consecutive frames
    // are spatially distinct (so the output isn't just N copies of one pose).
    const float az0  = cam.azimuth();
    const float step = (args.frames > 1)
        ? (6.2831853f / static_cast<float>(args.frames))
        : 0.0f;

    for (int i = 0; i < args.frames; ++i) {
        // Reset to deterministic pose for this frame index.
        cam.setOrbit(/*target*/ {0.f, 0.f, 0.f},
                     /*azimuth*/   az0 + step * static_cast<float>(i),
                     /*elevation*/ 0.35f,
                     /*distance*/  4.0f);
        renderer.dumpFrame(args.out, i);
    }
    std::printf("Done.\n");
    return 0;
}

int runWindowed(const Args& args) {
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) return 1;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
    glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(
        kInitialWidth, kInitialHeight, "Render-SR", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return 1;
    }

    int exit_code = 0;
    try {
        rs::Renderer renderer(window);
        renderer.setDebugBlit(args.debugBlit);
        std::printf("Metal device: %s\n", renderer.deviceName());

        double last = glfwGetTime();
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            const double now = glfwGetTime();
            const float  dt  = (float)(now - last);
            last = now;
            pollCameraInput(window, renderer.camera(), dt);
            renderer.drawFrame();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        exit_code = 1;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return exit_code;
}

}

int main(int argc, char** argv) {
    Args args;
    if (!parseArgs(argc, argv, args)) {
        printUsage(argv[0]);
        return 2;
    }
    try {
        return args.headless ? runHeadless(args) : runWindowed(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return 1;
    }
}
