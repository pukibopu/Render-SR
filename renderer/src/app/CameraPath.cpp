#include "CameraPath.hpp"

#include <cmath>

namespace rs {

namespace {

constexpr float kTwoPi = 6.28318530717958648f;

// splitmix64: a small, fast, well-distributed integer mixer. Used to turn an
// integer seed (+ a stream id, so independent quantities don't correlate) into
// a reproducible pseudo-random value with no global state.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Deterministic value in [0, 1) from a seed and a stream selector.
float seededUnit(std::uint32_t seed, std::uint32_t stream) {
    const std::uint64_t h =
        splitmix64((static_cast<std::uint64_t>(seed) << 32) ^ stream);
    return static_cast<float>(h >> 40) / static_cast<float>(1u << 24);
}

float lerp(float a, float b, float t) { return a + (b - a) * t; }

}

// Per-path seed derived from the base seed and the path id. Same base seed ->
// same per-path seeds -> same poses.
std::uint32_t derivePathSeed(std::uint32_t baseSeed, int pathId) {
    const std::uint64_t h =
        splitmix64((static_cast<std::uint64_t>(baseSeed) << 32) ^
                   static_cast<std::uint64_t>(pathId));
    return static_cast<std::uint32_t>(h & 0xFFFFFFFFull);
}

OrbitPath::OrbitPath(simd_float3 center, float radius, float height,
                     int frames, std::uint32_t seed)
    : CameraPath(frames, seed),
      m_center(center), m_radius(radius), m_height(height) {}

CameraPose OrbitPath::poseAt(int frameInPath) const {
    // t in [0, 1): a full loop across the frames so the first and last frame
    // are not duplicates.
    const float t = (m_frames > 0)
        ? static_cast<float>(frameInPath) / static_cast<float>(m_frames)
        : 0.0f;

    const float startPhase = seededUnit(m_seed, 1) * kTwoPi;
    const float azimuth    = startPhase + kTwoPi * t;

    const float baseElev = std::atan2(m_height, m_radius);
    const float wobbleAmp = (seededUnit(m_seed, 2) - 0.5f) * 0.15f;
    const float elevation = baseElev + wobbleAmp * std::sin(kTwoPi * t);

    const float distance =
        std::sqrt(m_radius * m_radius + m_height * m_height);

    return CameraPose{m_center, azimuth, elevation, distance};
}

DollyPath::DollyPath(CameraPose start, CameraPose end,
                     int frames, std::uint32_t seed)
    : CameraPath(frames, seed), m_start(start), m_end(end) {}

CameraPose DollyPath::poseAt(int frameInPath) const {
    // t in [0, 1] inclusive: a straight sweep from start to end.
    const float denom = (m_frames > 1)
        ? static_cast<float>(m_frames - 1) : 1.0f;
    const float t = static_cast<float>(frameInPath) / denom;

    const float drift =
        (seededUnit(m_seed, 1) - 0.5f) * 0.05f * std::sin(3.14159265f * t);

    CameraPose p;
    p.target = simd_make_float3(
        lerp(m_start.target.x, m_end.target.x, t),
        lerp(m_start.target.y, m_end.target.y, t),
        lerp(m_start.target.z, m_end.target.z, t));
    p.azimuth   = lerp(m_start.azimuth,   m_end.azimuth,   t) + drift;
    p.elevation = lerp(m_start.elevation, m_end.elevation, t);
    p.distance  = lerp(m_start.distance,  m_end.distance,  t);
    return p;
}

std::vector<PathEntry> makeDefaultPathSet(int framesPerPath,
                                          std::uint32_t baseSeed) {
    const simd_float3 origin = simd_make_float3(0.0f, 0.0f, 0.0f);

    std::vector<PathEntry> set;

    // Two orbits at different radius/height, two dollies in/out. The split
    // holds out one orbit and one dolly entirely for test, so evaluation never
    // sees a path it trained on.
    set.push_back(PathEntry{
        0, "train",
        std::make_unique<OrbitPath>(origin, 4.0f, 1.5f,
                                    framesPerPath, derivePathSeed(baseSeed, 0))});
    set.push_back(PathEntry{
        1, "test",
        std::make_unique<OrbitPath>(origin, 3.6f, 2.4f,
                                    framesPerPath, derivePathSeed(baseSeed, 1))});
    set.push_back(PathEntry{
        2, "train",
        std::make_unique<DollyPath>(
            CameraPose{origin,  0.6f, 0.30f, 5.5f},
            CameraPose{origin,  0.9f, 0.45f, 2.8f},
            framesPerPath, derivePathSeed(baseSeed, 2))});
    set.push_back(PathEntry{
        3, "test",
        std::make_unique<DollyPath>(
            CameraPose{origin, -0.8f, 0.50f, 5.0f},
            CameraPose{origin, -0.4f, 0.20f, 3.0f},
            framesPerPath, derivePathSeed(baseSeed, 3))});

    return set;
}

}
