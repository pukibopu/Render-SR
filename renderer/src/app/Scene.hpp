#pragma once

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

// A small, fixed, deterministic v1 scene: a few primitives at varied
// positions/scales/orientations, sharing 2-3 materials, lit by one directional
// light + ambient. Enough to produce real silhouettes, depth discontinuities,
// normal variation, and material edges for the SR experiment. No textures.
class Scene {
public:
    explicit Scene(MTL::Device* device);

    const std::vector<SceneObject>& objects() const { return m_objects; }

    // Directional light travel direction, world space (points from light).
    simd_float3 lightDirWorld() const { return m_lightDirWorld; }
    float       ambient()       const { return m_ambient; }

private:
    std::vector<std::unique_ptr<Mesh>> m_meshes;   // GPU buffers, kept alive
    std::vector<SceneObject>           m_objects;
    simd_float3 m_lightDirWorld{0.0f, -1.0f, 0.0f};
    float       m_ambient = 0.15f;
};

}
