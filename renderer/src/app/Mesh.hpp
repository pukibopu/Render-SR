#pragma once

#include <cstdint>
#include <vector>

namespace MTL { class Buffer; class Device; }

namespace rs {

struct MeshVertex {
    float position[3];
    float normal[3];
    float uv[2];
};
static_assert(sizeof(MeshVertex) == 32, "MeshVertex must be 32 bytes (tightly packed)");

struct MeshCPU {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t>   indices;
};

namespace primitives {

MeshCPU cube();
MeshCPU plane(float size);
MeshCPU uvSphere(uint32_t stacks, uint32_t slices, float radius);
MeshCPU torus(uint32_t ringSegments, uint32_t tubeSegments,
              float majorRadius, float minorRadius);

}

class Mesh {
public:
    Mesh(MTL::Device* device, const MeshCPU& cpu);
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    MTL::Buffer* vertexBuffer() const { return m_vbuf; }
    MTL::Buffer* indexBuffer()  const { return m_ibuf; }
    uint32_t     indexCount()   const { return m_indexCount; }

private:
    MTL::Buffer* m_vbuf = nullptr;
    MTL::Buffer* m_ibuf = nullptr;
    uint32_t     m_indexCount = 0;
};

}
