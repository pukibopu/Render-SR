#pragma once

#include <simd/simd.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rs {

// A camera pose expressed in the orbit-camera parameters that Camera::setOrbit
// consumes, so a path can drive the existing camera directly.
struct CameraPose {
    simd_float3 target;
    float       azimuth;
    float       elevation;
    float       distance;
};

// Deterministic camera path. poseAt(i) is a pure function of i, the path
// parameters, and the seed — so it is identical across runs and independent of
// the order frames are evaluated in. This is what makes the dataset
// reproducible from a single --seed.
class CameraPath {
public:
    virtual ~CameraPath() = default;

    virtual CameraPose  poseAt(int frameInPath) const = 0;
    virtual const char* type() const = 0;

    int           frameCount() const { return m_frames; }
    std::uint32_t seed()       const { return m_seed; }

protected:
    CameraPath(int frames, std::uint32_t seed)
        : m_frames(frames), m_seed(seed) {}

    int           m_frames;
    std::uint32_t m_seed;
};

// Full-circle orbit around `center` at fixed radius/height. The seed sets the
// start phase and a small elevation wobble so different orbit paths are
// distinct but each is reproducible.
class OrbitPath : public CameraPath {
public:
    OrbitPath(simd_float3 center, float radius, float height,
              int frames, std::uint32_t seed);

    CameraPose  poseAt(int frameInPath) const override;
    const char* type() const override { return "orbit"; }

private:
    simd_float3 m_center;
    float       m_radius;
    float       m_height;
};

// Linear sweep between two poses (a dolly in/out plus orientation change). The
// seed adds a tiny deterministic azimuth drift so dolly paths differ.
class DollyPath : public CameraPath {
public:
    DollyPath(CameraPose start, CameraPose end,
              int frames, std::uint32_t seed);

    CameraPose  poseAt(int frameInPath) const override;
    const char* type() const override { return "dolly"; }

private:
    CameraPose m_start;
    CameraPose m_end;
};

// One entry in a path set: a path plus its dataset role. The train/test split
// is recorded here, per path, so the dataset side can split by path ID rather
// than by frame (adjacent frames in a path are near duplicates; splitting on
// them leaks). `scene_variant` selects a deterministic scene perturbation for
// the path (0 = the fixed base scene; non-zero = seeded transform/light jitter).
struct PathEntry {
    int                          id;
    std::string                  split;        // "train" | "test"
    std::uint32_t                scene_variant; // 0 = base scene, else seeded
    std::unique_ptr<CameraPath>  path;
};

// Build the hardcoded v1 path set: a small fixed list of orbits and dollies on
// the fixed base scene (scene_variant 0). Each path's seed is derived
// deterministically from `baseSeed` and the path id, so a single --seed
// reproduces the entire set. `framesPerPath` sets every path's frame count.
std::vector<PathEntry> makeDefaultPathSet(int framesPerPath, std::uint32_t baseSeed);

// Build a larger, more diverse v2 path set: `numPaths` procedurally generated
// paths (a deterministic mix of orbits and dollies with seeded radius/height/
// target/elevation spread), each carrying its own non-zero `scene_variant` so
// object transforms and lighting vary per path. The split is by path id (every
// 4th path held out for test). Fully reproducible from `baseSeed`.
std::vector<PathEntry> makeDatasetV2PathSet(int numPaths, int framesPerPath,
                                            std::uint32_t baseSeed);

// Per-path seed derived from the base seed and the path id (exposed for reuse).
std::uint32_t derivePathSeed(std::uint32_t baseSeed, int pathId);

}
