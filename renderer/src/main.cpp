#include "app/Renderer.hpp"
#include "app/CameraPath.hpp"
#include "io/FrameWriter.hpp"

#include <GLFW/glfw3.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr int kInitialWidth  = 1280;
constexpr int kInitialHeight = 720;

struct Args {
    bool headless = false;
    std::string   paths         = "default";  // "default" (v1) | "v2"
    int           framesPerPath = 8;
    int           numPaths      = 16;          // v2 only
    std::uint32_t seed          = 42;
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
        "       %s --headless [--paths {default|v2}] [--num-paths N]\n"
        "                       [--frames-per-path K] [--seed S] [--out PATH]\n"
        "\n"
        "  --debug-blit       which low-res G-buffer attachment to display on\n"
        "                     screen (defaults to rgb; ignored in headless mode)\n"
        "  --headless         render without a window; write frames to PATH\n"
        "  --paths NAME       camera path set: 'default' (v1: 4 fixed paths, fixed\n"
        "                     scene) or 'v2' (N procedural paths, per-path scene\n"
        "                     variants). Default 'default'.\n"
        "  --num-paths N      number of paths for --paths v2 (default 16)\n"
        "  --frames-per-path K  frames rendered per camera path (default 8)\n"
        "  --frames K         deprecated alias for --frames-per-path\n"
        "  --seed S           base seed; same seed reproduces the dataset\n"
        "                     (default 42)\n"
        "  --out PATH         output directory root (defaults to ./output_buffers)\n"
        "\n"
        "  dataset v2 (~1024 frames): --headless --paths v2 --num-paths 16 \\\n"
        "                             --frames-per-path 64 --seed 42 --out output_buffers_v2\n",
        argv0, argv0);
}

bool parseUInt(const char* v, const char* flag, std::uint32_t& out) {
    // strtoul silently wraps a leading '-'; reject it explicitly.
    if (v[0] == '-' || v[0] == '+' || v[0] == '\0') {
        std::fprintf(stderr, "error: %s expects a non-negative integer, got '%s'\n", flag, v);
        return false;
    }
    char* end = nullptr;
    unsigned long n = std::strtoul(v, &end, 10);
    if (!end || *end != '\0') {
        std::fprintf(stderr, "error: %s expects a non-negative integer, got '%s'\n", flag, v);
        return false;
    }
    out = static_cast<std::uint32_t>(n);
    return true;
}

bool parsePositiveInt(const char* v, const char* flag, int& out) {
    char* end = nullptr;
    long n = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || n <= 0) {
        std::fprintf(stderr, "error: %s expects a positive integer, got '%s'\n", flag, v);
        return false;
    }
    out = static_cast<int>(n);
    return true;
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
        } else if (a == "--paths") {
            const char* v = next("--paths");
            if (!v) return false;
            if (std::strcmp(v, "default") != 0 && std::strcmp(v, "v2") != 0) {
                std::fprintf(stderr,
                    "error: --paths supports 'default' or 'v2', got '%s'\n", v);
                return false;
            }
            out_args.paths = v;
        } else if (a == "--num-paths") {
            const char* v = next("--num-paths");
            if (!v) return false;
            if (!parsePositiveInt(v, "--num-paths", out_args.numPaths)) return false;
        } else if (a == "--frames-per-path" || a == "--frames") {
            const char* v = next(a.c_str());
            if (!v) return false;
            if (!parsePositiveInt(v, "--frames-per-path", out_args.framesPerPath)) return false;
        } else if (a == "--seed") {
            const char* v = next("--seed");
            if (!v) return false;
            if (!parseUInt(v, "--seed", out_args.seed)) return false;
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
    return true;
}

int runHeadless(const Args& args) {
    rs::Renderer renderer(nullptr);
    std::printf("Metal device: %s\n", renderer.deviceName());

    const bool isV2 = (args.paths == "v2");
    const int datasetVersion = isV2 ? 2 : 1;
    std::vector<rs::PathEntry> paths =
        isV2 ? rs::makeDatasetV2PathSet(args.numPaths, args.framesPerPath, args.seed)
             : rs::makeDefaultPathSet(args.framesPerPath, args.seed);

    const int totalFrames =
        static_cast<int>(paths.size()) * args.framesPerPath;
    std::printf("Path set '%s' (dataset v%d): %zu paths x %d frames = %d frame%s "
                "(seed %u) under: %s\n",
                args.paths.c_str(), datasetVersion, paths.size(),
                args.framesPerPath, totalFrames, totalFrames == 1 ? "" : "s",
                args.seed, args.out.string().c_str());

    rs::Camera& cam = renderer.camera();

    std::vector<rs::io::PathSummary>  pathSummaries;
    std::vector<rs::io::ManifestEntry> manifest;
    pathSummaries.reserve(paths.size());
    manifest.reserve(totalFrames);

    for (const rs::PathEntry& entry : paths) {
        const rs::CameraPath& path = *entry.path;
        pathSummaries.push_back(rs::io::PathSummary{
            entry.id, path.type(), entry.split, path.seed(),
            path.frameCount(), entry.scene_variant});

        // Apply this path's scene variant once before rendering its frames.
        renderer.setSceneVariant(entry.scene_variant);

        for (int f = 0; f < path.frameCount(); ++f) {
            const rs::CameraPose pose = path.poseAt(f);
            cam.setOrbit(pose.target, pose.azimuth, pose.elevation, pose.distance);
            rs::io::CameraSnapshot snap = renderer.dumpFrame(args.out, entry.id, f);
            manifest.push_back(rs::io::ManifestEntry{
                entry.id, path.type(), entry.split, path.seed(), f, snap});
        }
    }

    rs::io::writeManifest(args.out, datasetVersion, args.seed, args.framesPerPath,
                          renderer.lowResWidth(),  renderer.lowResHeight(),
                          renderer.highResWidth(), renderer.highResHeight(),
                          pathSummaries, manifest);

    std::printf("Wrote %d frames + manifest.json. Done.\n", totalFrames);
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
