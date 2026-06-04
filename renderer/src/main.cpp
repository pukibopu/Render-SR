#include "app/Renderer.hpp"

#include <GLFW/glfw3.h>

#include <cstdio>
#include <exception>

namespace {

constexpr int kInitialWidth  = 1280;
constexpr int kInitialHeight = 720;

void glfwErrorCallback(int code, const char* description) {
    std::fprintf(stderr, "glfw error %d: %s\n", code, description);
}

}

int main() {
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
        std::printf("Metal device: %s\n", renderer.deviceName());

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
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
