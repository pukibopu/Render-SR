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

    // Ground plane (large, flat) — silhouettes + depth backdrop.
    m_objects.push_back({mPlane,
        simd_mul(translation(0.f, -0.9f, 0.f), scaling(9.f, 1.f, 9.f)),
        matFloor});

    // Center torus, tilted — strong curvature for normal variation.
    m_objects.push_back({mTorus,
        simd_mul(rotationX(0.55f), scaling(0.95f, 0.95f, 0.95f)),
        matWarm});

    // Left cube, rotated — flat faces + hard material edges.
    m_objects.push_back({mCube,
        simd_mul(simd_mul(translation(-1.9f, -0.35f, 0.4f), rotationY(0.7f)),
                 scaling(0.6f, 0.6f, 0.6f)),
        matCool});

    // Right sphere — clean curved silhouette at a different depth.
    m_objects.push_back({mSphere,
        simd_mul(translation(1.8f, -0.25f, -0.6f), scaling(0.65f, 0.65f, 0.65f)),
        matWarm});

    // Small tall cube behind — depth discontinuity against the others.
    m_objects.push_back({mCube,
        simd_mul(simd_mul(translation(0.3f, -0.2f, -2.0f), rotationY(-0.4f)),
                 scaling(0.45f, 0.9f, 0.45f)),
        matCool});

    // Small sphere foreground — overlap/occlusion edges.
    m_objects.push_back({mSphere,
        simd_mul(translation(-0.8f, -0.55f, 1.6f), scaling(0.35f, 0.35f, 0.35f)),
        matFloor});

    // One directional light (travels down-forward-right) + ambient.
    m_lightDirWorld = simd_normalize(simd_make_float3(-0.4f, -0.85f, -0.5f));
    m_ambient = 0.16f;
}

}
