#include "FrameWriter.hpp"
#include "NpyWriter.hpp"
#include "StbWriter.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace rs::io {

namespace {

std::string fmtF(float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return std::string(buf);
}

std::string fmtVec3(simd_float3 v) {
    return "[" + fmtF(v.x) + ", " + fmtF(v.y) + ", " + fmtF(v.z) + "]";
}

// Row-major (mathematical) flattening of a column-major simd matrix.
std::string fmtMat4RowMajor(simd_float4x4 m) {
    std::string s = "[";
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (r != 0 || c != 0) s += ", ";
            s += fmtF(m.columns[c][r]);
        }
    }
    s += "]";
    return s;
}

void writeMeta(const std::filesystem::path& path,
               int pathId, int frameInPath,
               std::uint32_t lowW,  std::uint32_t lowH,
               std::uint32_t highW, std::uint32_t highH,
               const CameraSnapshot& cam)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("FrameWriter: failed to open " + path.string());

    out << "{\n";
    out << "  \"path_id\":   " << pathId      << ",\n";
    out << "  \"frame_idx\": " << frameInPath << ",\n";
    out << "  \"low_res\":  [" << lowW  << ", " << lowH  << "],\n";
    out << "  \"high_res\": [" << highW << ", " << highH << "],\n";
    out << "  \"camera\": {\n";
    out << "    \"fov_y_rad\":   " << fmtF(cam.fov_y)      << ",\n";
    out << "    \"aspect\":      " << fmtF(cam.aspect)     << ",\n";
    out << "    \"near\":        " << fmtF(cam.near_plane) << ",\n";
    out << "    \"far\":         " << fmtF(cam.far_plane)  << ",\n";
    out << "    \"target\":      " << fmtVec3(cam.target)  << ",\n";
    out << "    \"azimuth_rad\": " << fmtF(cam.azimuth)    << ",\n";
    out << "    \"elevation_rad\": " << fmtF(cam.elevation) << ",\n";
    out << "    \"distance\":    " << fmtF(cam.distance)   << ",\n";
    out << "    \"eye\":         " << fmtVec3(cam.eye)     << ",\n";
    out << "    \"view_row_major\": " << fmtMat4RowMajor(cam.view) << "\n";
    out << "  },\n";
    out << "  \"depth\": {\n";
    out << "    \"format\": \"float32\",\n";
    out << "    \"convention\": \"linear_eye_space_minus_viewZ_positive_into_scene\",\n";
    out << "    \"shape\": [" << lowH << ", " << lowW << "]\n";
    out << "  },\n";
    out << "  \"normal\": {\n";
    out << "    \"format\": \"float32\",\n";
    out << "    \"convention\": \"view_space_right_handed_minus_z_forward\",\n";
    out << "    \"components_range\": [-1.0, 1.0],\n";
    out << "    \"shape\": [" << lowH << ", " << lowW << ", 3]\n";
    out << "  },\n";
    out << "  \"color\": {\n";
    out << "    \"format\": \"png_uint8_rgb\",\n";
    out << "    \"encoding\": \"sRGB\"\n";
    out << "  }\n";
    out << "}\n";

    if (!out) throw std::runtime_error("FrameWriter: write failed for " + path.string());
}

}

std::string frameStem(int pathId, int frameInPath) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "frame_%02d_%04d", pathId, frameInPath);
    return std::string(buf);
}

void writeFrame(const std::filesystem::path& outRoot,
                int pathId, int frameInPath,
                std::uint32_t lowW,  std::uint32_t lowH,
                std::uint32_t highW, std::uint32_t highH,
                const std::uint8_t* lowRGBA,
                const float*        lowDepth,
                const float*        lowNormal3,
                const std::uint8_t* highRGBA,
                const CameraSnapshot& cam)
{
    const std::filesystem::path rgbLowDir  = outRoot / "rgb_low";
    const std::filesystem::path rgbHighDir = outRoot / "rgb_high";
    const std::filesystem::path depthDir   = outRoot / "depth";
    const std::filesystem::path normalDir  = outRoot / "normal";
    const std::filesystem::path metaDir    = outRoot / "meta";

    std::filesystem::create_directories(rgbLowDir);
    std::filesystem::create_directories(rgbHighDir);
    std::filesystem::create_directories(depthDir);
    std::filesystem::create_directories(normalDir);
    std::filesystem::create_directories(metaDir);

    const std::string stem = frameStem(pathId, frameInPath);

    if (!writePngRGBfromRGBA8(rgbLowDir / (stem + ".png"),
                              lowRGBA, (int)lowW, (int)lowH)) {
        throw std::runtime_error("FrameWriter: PNG write failed for rgb_low/" + stem);
    }
    if (!writePngRGBfromRGBA8(rgbHighDir / (stem + ".png"),
                              highRGBA, (int)highW, (int)highH)) {
        throw std::runtime_error("FrameWriter: PNG write failed for rgb_high/" + stem);
    }

    writeNpyF32(depthDir  / (stem + ".npy"), lowDepth,    {lowH, lowW});
    writeNpyF32(normalDir / (stem + ".npy"), lowNormal3,  {lowH, lowW, 3});

    writeMeta(metaDir / (stem + ".json"),
              pathId, frameInPath, lowW, lowH, highW, highH, cam);
}

void writeManifest(const std::filesystem::path& outRoot,
                   std::uint32_t baseSeed, int framesPerPath,
                   std::uint32_t lowW,  std::uint32_t lowH,
                   std::uint32_t highW, std::uint32_t highH,
                   const std::vector<PathSummary>&  paths,
                   const std::vector<ManifestEntry>& frames)
{
    std::filesystem::create_directories(outRoot);
    const std::filesystem::path path = outRoot / "manifest.json";

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("FrameWriter: failed to open " + path.string());

    out << "{\n";
    out << "  \"seed\": "            << baseSeed       << ",\n";
    out << "  \"frames_per_path\": " << framesPerPath  << ",\n";
    out << "  \"low_res\":  [" << lowW  << ", " << lowH  << "],\n";
    out << "  \"high_res\": [" << highW << ", " << highH << "],\n";

    out << "  \"paths\": [\n";
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const PathSummary& p = paths[i];
        out << "    {\"path_id\": " << p.id
            << ", \"type\": \""     << p.type  << "\""
            << ", \"split\": \""    << p.split << "\""
            << ", \"seed\": "       << p.seed
            << ", \"frames\": "     << p.frames << "}"
            << (i + 1 < paths.size() ? "," : "") << "\n";
    }
    out << "  ],\n";

    out << "  \"frames\": [\n";
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const ManifestEntry& e = frames[i];
        const std::string stem = frameStem(e.path_id, e.frame_in_path);
        out << "    {\"frame\": \""  << stem << "\""
            << ", \"path_id\": "      << e.path_id
            << ", \"path_type\": \""  << e.path_type << "\""
            << ", \"split\": \""      << e.split     << "\""
            << ", \"seed\": "         << e.seed
            << ", \"frame_in_path\": " << e.frame_in_path
            << ", \"camera\": {"
            <<   "\"target\": "      << fmtVec3(e.cam.target)
            << ", \"azimuth_rad\": " << fmtF(e.cam.azimuth)
            << ", \"elevation_rad\": " << fmtF(e.cam.elevation)
            << ", \"distance\": "    << fmtF(e.cam.distance)
            << ", \"eye\": "         << fmtVec3(e.cam.eye)
            << "}}"
            << (i + 1 < frames.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    if (!out) throw std::runtime_error("FrameWriter: write failed for " + path.string());
}

}
