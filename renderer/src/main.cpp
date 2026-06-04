#include "app/Renderer.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <string>

namespace {

constexpr int kInitialWidth  = 1280;
constexpr int kInitialHeight = 720;

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
        "  --debug-blit  which low-res G-buffer attachment to display on screen\n"
        "                (defaults to rgb)\n",
        argv0);
}

}

int main(int argc, char** argv) {
    rs::GBufferPass::DebugBlit debugBlit = rs::GBufferPass::DebugBlit::RGB;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--debug-blit") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --debug-blit requires an argument\n");
                printUsage(argv[0]);
                return 2;
            }
            auto parsed = parseDebugBlit(argv[++i]);
            if (!parsed) {
                std::fprintf(stderr,
                    "error: --debug-blit expects rgb|depth|normal, got '%s'\n", argv[i]);
                return 2;
            }
            debugBlit = *parsed;
        } else if (a == "--help" || a == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            printUsage(argv[0]);
            return 2;
        }
    }

    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        return 1;
    }

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
        renderer.setDebugBlit(debugBlit);
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
