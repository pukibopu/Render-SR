#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
#include <simd/simd.h>

namespace rs::io {

struct CameraSnapshot {
    float fov_y;
    float aspect;
    float near_plane;
    float far_plane;
    simd_float3 target;
    float azimuth;
    float elevation;
    float distance;
    simd_float3 eye;
    simd_float4x4 view;
};

// Frame file stem: "frame_PP_FFFF" — PP = path id, FFFF = frame within path.
// The path id is part of the name so the dataset side can split by path.
std::string frameStem(int pathId, int frameInPath);

// Write one frame's outputs under outRoot:
//   outRoot/rgb_low/frame_PP_FFFF.png   (uint8 RGB,  width=lowW,  height=lowH)
//   outRoot/rgb_high/frame_PP_FFFF.png  (uint8 RGB,  width=highW, height=highH)
//   outRoot/depth/frame_PP_FFFF.npy     (float32,    shape=(lowH, lowW))
//   outRoot/normal/frame_PP_FFFF.npy    (float32,    shape=(lowH, lowW, 3),
//                                        components in [-1, 1])
//   outRoot/meta/frame_PP_FFFF.json     (path_id, frame_idx, camera, conventions)
//
// Inputs are tightly packed, row-major. `lowRGBA` and `highRGBA` are RGBA8
// (sRGB-encoded). `lowDepth` is linear eye-space depth = -viewPos.z.
// `lowNormal3` is view-space normal, alpha already dropped — three float32
// components per pixel in [-1, 1].
void writeFrame(const std::filesystem::path& outRoot,
                int pathId, int frameInPath,
                std::uint32_t lowW,  std::uint32_t lowH,
                std::uint32_t highW, std::uint32_t highH,
                const std::uint8_t* lowRGBA,
                const float*        lowDepth,
                const float*        lowNormal3,
                const std::uint8_t* highRGBA,
                const CameraSnapshot& cam);

// One row of the dataset manifest: which path a frame belongs to, its role in
// the train/test split, the path's seed, and the camera snapshot at that frame.
struct ManifestEntry {
    int           path_id;
    std::string   path_type;   // "orbit" | "dolly"
    std::string   split;       // "train" | "test"
    std::uint32_t seed;        // the path's derived seed
    int           frame_in_path;
    CameraSnapshot cam;
};

// One row of the per-path summary in the manifest header.
struct PathSummary {
    int           id;
    std::string   type;
    std::string   split;
    std::uint32_t seed;
    int           frames;
};

// Write outRoot/manifest.json: the run parameters, a per-path summary, and one
// record per frame keyed by (path_id, frame_in_path). The dataset loader keys
// off this to split train/test by path id, never by frame.
void writeManifest(const std::filesystem::path& outRoot,
                   std::uint32_t baseSeed, int framesPerPath,
                   std::uint32_t lowW,  std::uint32_t lowH,
                   std::uint32_t highW, std::uint32_t highH,
                   const std::vector<PathSummary>&  paths,
                   const std::vector<ManifestEntry>& frames);

}
