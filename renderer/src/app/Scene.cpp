#include "Scene.hpp"
#include "Mesh.hpp"

#include <cmath>

namespace rs {

namespace {

simd_float4x4 translation(float x, float y, float z) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[3] = simd_make_float4(x, y, z, 1.0f);
    return m;
}

simd_float4x4 scaling(float sx, float sy, float sz) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0].x = sx;
    m.columns[1].y = sy;
    m.columns[2].z = sz;
    return m;
}

simd_float4x4 rotationY(float a) {
    const float c = std::cos(a), s = std::sin(a);
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0] = simd_make_float4( c, 0.f, -s, 0.f);
    m.columns[2] = simd_make_float4( s, 0.f,  c, 0.f);
    return m;
}

simd_float4x4 rotationX(float a) {
    const float c = std::cos(a), s = std::sin(a);
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[1] = simd_make_float4(0.f,  c, s, 0.f);
    m.columns[2] = simd_make_float4(0.f, -s, c, 0.f);
    return m;
}

// splitmix64 + seeded unit value: the same deterministic mixer the camera
// paths use, so scene variants are reproducible from a single seed.
std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

float seededUnit(std::uint32_t seed, std::uint32_t stream) {
    const std::uint64_t h =
        splitmix64((static_cast<std::uint64_t>(seed) << 32) ^ stream);
    return static_cast<float>(h >> 40) / static_cast<float>(1u << 24);
}

float seededRange(std::uint32_t seed, std::uint32_t stream, float lo, float hi) {
    return lo + (hi - lo) * seededUnit(seed, stream);
}

}

Scene::Scene(MTL::Device* device) {
    // Shared meshes (reused across objects to keep buffers minimal).
    auto plane  = std::make_unique<Mesh>(device, primitives::plane(1.0f));
    auto cube   = std::make_unique<Mesh>(device, primitives::cube());
    auto sphere = std::make_unique<Mesh>(device, primitives::uvSphere(32, 48, 1.0f));
    auto torus  = std::make_unique<Mesh>(device, primitives::torus(48, 32, 1.0f, 0.32f));

    const Mesh* mPlane  = plane.get();
    const Mesh* mCube   = cube.get();
    const Mesh* mSphere = sphere.get();
    const Mesh* mTorus  = torus.get();
    m_meshes.push_back(std::move(plane));
    m_meshes.push_back(std::move(cube));
    m_meshes.push_back(std::move(sphere));
    m_meshes.push_back(std::move(torus));

    // Three materials, reused: matte gray floor, warm semi-gloss, cool glossy.
    const Material matFloor { simd_make_float3(0.55f, 0.55f, 0.58f), 0.05f };
    const Material matWarm  { simd_make_float3(0.78f, 0.55f, 0.32f), 0.35f };
    const Material matCool  { simd_make_float3(0.30f, 0.45f, 0.70f), 0.75f };

    // Decomposed base layout (variant 0). World matrix = T * Ry * Rx * S, which
    // reproduces the original v1 composition exactly for every object. `jitter`
    // is false only for the ground plane (kept as a stable depth backdrop).
    m_base = {
        // Ground plane (large, flat) — silhouettes + depth backdrop.
        {mPlane,  {0.f, -0.9f, 0.f},   0.0f,  0.0f, {9.f, 1.f, 9.f},
         matFloor, false},
        // Center torus, tilted — strong curvature for normal variation.
        {mTorus,  {0.f, 0.f, 0.f},     0.0f,  0.55f, {0.95f, 0.95f, 0.95f},
         matWarm, true},
        // Left cube, rotated — flat faces + hard material edges.
        {mCube,   {-1.9f, -0.35f, 0.4f}, 0.7f, 0.0f, {0.6f, 0.6f, 0.6f},
         matCool, true},
        // Right sphere — clean curved silhouette at a different depth.
        {mSphere, {1.8f, -0.25f, -0.6f}, 0.0f, 0.0f, {0.65f, 0.65f, 0.65f},
         matWarm, true},
        // Small tall cube behind — depth discontinuity against the others.
        {mCube,   {0.3f, -0.2f, -2.0f}, -0.4f, 0.0f, {0.45f, 0.9f, 0.45f},
         matCool, true},
        // Small sphere foreground — overlap/occlusion edges.
        {mSphere, {-0.8f, -0.55f, 1.6f}, 0.0f, 0.0f, {0.35f, 0.35f, 0.35f},
         matFloor, true},
    };

    // One directional light (travels down-forward-right) + ambient.
    m_baseLightDirWorld = simd_normalize(simd_make_float3(-0.4f, -0.85f, -0.5f));
    m_ambient = 0.16f;

    setVariant(0);   // build the base world list
}

void Scene::setVariant(std::uint32_t seed) {
    m_objects.clear();
    m_objects.reserve(m_base.size());

    for (std::size_t i = 0; i < m_base.size(); ++i) {
        const BaseObject& b = m_base[i];

        simd_float3 t = b.translate;
        float       ry = b.rotY;
        float       rx = b.rotX;
        simd_float3 s  = b.scale;

        if (seed != 0u && b.jitter) {
            // Per-object stream offset so objects don't move in lockstep. Bounds
            // keep objects roughly within the camera framing while spreading
            // them in depth (z) and screen-x for more occlusion/depth variation.
            const std::uint32_t o = static_cast<std::uint32_t>(i) * 16u;
            t.x += seededRange(seed, o + 1, -0.9f, 0.9f);
            t.y += seededRange(seed, o + 2, -0.12f, 0.12f);
            t.z += seededRange(seed, o + 3, -1.4f, 1.4f);
            ry  += seededRange(seed, o + 4, -3.14159265f, 3.14159265f);
            rx  += seededRange(seed, o + 5, -0.4f, 0.4f);
            const float sf = seededRange(seed, o + 6, 0.72f, 1.30f);
            s = simd_make_float3(s.x * sf, s.y * sf, s.z * sf);
        }

        const simd_float4x4 model = simd_mul(
            simd_mul(translation(t.x, t.y, t.z),
                     simd_mul(rotationY(ry), rotationX(rx))),
            scaling(s.x, s.y, s.z));
        m_objects.push_back({b.mesh, model, b.material});
    }

    // Jitter the light azimuth a little per variant for shading variation; the
    // base direction is restored exactly for variant 0.
    if (seed == 0u) {
        m_lightDirWorld = m_baseLightDirWorld;
    } else {
        const float a = seededRange(seed, 9991, -0.6f, 0.6f);
        const float c = std::cos(a), sn = std::sin(a);
        const simd_float3 d = m_baseLightDirWorld;
        m_lightDirWorld = simd_normalize(simd_make_float3(
            c * d.x + sn * d.z, d.y, -sn * d.x + c * d.z));
    }
}

}
