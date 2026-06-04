#pragma once

#include <cstdint>
#include <filesystem>
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

// Write one frame's outputs under outRoot:
//   outRoot/rgb_low/frame_NNNN.png   (uint8 RGB,  width=lowW,  height=lowH)
//   outRoot/rgb_high/frame_NNNN.png  (uint8 RGB,  width=highW, height=highH)
//   outRoot/depth/frame_NNNN.npy     (float32,    shape=(lowH, lowW))
//   outRoot/normal/frame_NNNN.npy    (float32,    shape=(lowH, lowW, 3),
//                                     components in [-1, 1])
//   outRoot/meta/frame_NNNN.json     (camera pose + conventions)
//
// Inputs are tightly packed, row-major. `lowRGBA` and `highRGBA` are RGBA8
// (sRGB-encoded). `lowDepth` is linear eye-space depth = -viewPos.z.
// `lowNormal3` is view-space normal, alpha already dropped — three float32
// components per pixel in [-1, 1].
void writeFrame(const std::filesystem::path& outRoot,
                int frameIndex,
                std::uint32_t lowW,  std::uint32_t lowH,
                std::uint32_t highW, std::uint32_t highH,
                const std::uint8_t* lowRGBA,
                const float*        lowDepth,
                const float*        lowNormal3,
                const std::uint8_t* highRGBA,
                const CameraSnapshot& cam);

}
