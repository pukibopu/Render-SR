#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <simd/simd.h>

namespace MTL { class Device; }

namespace rs {

class Mesh;

struct Material {
    simd_float3 albedo;
    float       specularStrength;
};

struct SceneObject {
    const Mesh*   mesh;   // owned by Scene (not this struct)
    simd_float4x4 model;
    Material      material;
};

// A small, deterministic scene: a few primitives at varied positions/scales/
// orientations, sharing 2-3 materials, lit by one directional light + ambient.
// Enough to produce real silhouettes, depth discontinuities, normal variation,
// and material edges for the SR experiment. No textures, no PBR.
//
// The object layout is stored as decomposed base transforms; setVariant(seed)
// rebuilds the world transforms (and jitters the light) from a deterministic
// per-variant perturbation. setVariant(0) reproduces the fixed base layout
// exactly, so the v1 dataset is unchanged; non-zero variants spread objects in
// depth/screen space for more occlusion and per-path diversity (dataset v2).
class Scene {
public:
    explicit Scene(MTL::Device* device);

    // Select a deterministic scene variant (0 = fixed base layout). Rebuilds
    // the object world matrices and the light direction; meshes/materials are
    // reused. Calling with the same seed always yields the same scene.
    void setVariant(std::uint32_t seed);

    const std::vector<SceneObject>& objects() const { return m_objects; }

    // Directional light travel direction, world space (points from light).
    simd_float3 lightDirWorld() const { return m_lightDirWorld; }
    float       ambient()       const { return m_ambient; }

private:
    // One object's base (variant-0) transform, stored decomposed so a variant
    // can perturb translation/rotation/scale before composing the matrix.
    struct BaseObject {
        const Mesh*  mesh;
        simd_float3  translate;
        float        rotY;
        float        rotX;
        simd_float3  scale;
        Material     material;
        bool         jitter;   // false for the ground plane (stable backdrop)
    };

    std::vector<std::unique_ptr<Mesh>> m_meshes;   // GPU buffers, kept alive
    std::vector<BaseObject>            m_base;      // decomposed base layout
    std::vector<SceneObject>           m_objects;   // current (variant) world list
    simd_float3 m_baseLightDirWorld{0.0f, -1.0f, 0.0f};
    simd_float3 m_lightDirWorld{0.0f, -1.0f, 0.0f};
    float       m_ambient = 0.15f;
};

}
