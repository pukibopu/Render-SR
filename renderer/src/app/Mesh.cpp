#include "Mesh.hpp"

#include <Metal/Metal.hpp>

#include <cmath>
#include <stdexcept>

namespace rs {

Mesh::Mesh(MTL::Device* device, const MeshCPU& cpu) {
    if (cpu.vertices.empty() || cpu.indices.empty()) {
        throw std::runtime_error("Mesh: empty CPU data");
    }
    m_vbuf = device->newBuffer(
        cpu.vertices.data(),
        cpu.vertices.size() * sizeof(MeshVertex),
        MTL::ResourceStorageModeShared);
    m_ibuf = device->newBuffer(
        cpu.indices.data(),
        cpu.indices.size() * sizeof(uint32_t),
        MTL::ResourceStorageModeShared);
    if (!m_vbuf || !m_ibuf) {
        throw std::runtime_error("Mesh: failed to allocate buffer(s)");
    }
    m_indexCount = (uint32_t)cpu.indices.size();
}

Mesh::~Mesh() {
    if (m_vbuf) m_vbuf->release();
    if (m_ibuf) m_ibuf->release();
}

namespace primitives {

namespace {

struct FaceAxes { float n[3], u[3], v[3]; };

void appendQuad(MeshCPU& out, const FaceAxes& f) {
    const uint32_t base = (uint32_t)out.vertices.size();
    auto vert = [&](float su, float sv, float tu, float tv) {
        MeshVertex mv;
        mv.position[0] = 0.5f * (f.n[0] + su * f.u[0] + sv * f.v[0]);
        mv.position[1] = 0.5f * (f.n[1] + su * f.u[1] + sv * f.v[1]);
        mv.position[2] = 0.5f * (f.n[2] + su * f.u[2] + sv * f.v[2]);
        mv.normal[0] = f.n[0]; mv.normal[1] = f.n[1]; mv.normal[2] = f.n[2];
        mv.uv[0] = tu; mv.uv[1] = tv;
        out.vertices.push_back(mv);
    };
    vert(-1.f, -1.f, 0.f, 0.f);
    vert( 1.f, -1.f, 1.f, 0.f);
    vert( 1.f,  1.f, 1.f, 1.f);
    vert(-1.f,  1.f, 0.f, 1.f);
    out.indices.insert(out.indices.end(),
        {base + 0, base + 1, base + 2, base + 0, base + 2, base + 3});
}

}

MeshCPU cube() {
    MeshCPU m;
    m.vertices.reserve(24);
    m.indices.reserve(36);
    // 6 faces, each with consistent CCW winding from outside.
    // For each face: normal n, and two in-plane axes (u, v) chosen so that
    // n = u × v (right-handed) — this fixes CCW order from the outside.
    const FaceAxes faces[6] = {
        // +X: u = +Z, v = +Y  (cross(+Z,+Y) = -X — flip to make outside CCW)
        {{ 1, 0, 0}, {0, 0,-1}, {0, 1, 0}},
        // -X
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},
        // +Y
        {{ 0, 1, 0}, {1, 0, 0}, {0, 0, 1}},
        // -Y
        {{ 0,-1, 0}, {1, 0, 0}, {0, 0,-1}},
        // +Z
        {{ 0, 0, 1}, {1, 0, 0}, {0, 1, 0}},
        // -Z
        {{ 0, 0,-1}, {-1,0, 0}, {0, 1, 0}},
    };
    for (const auto& f : faces) appendQuad(m, f);
    return m;
}

MeshCPU plane(float size) {
    MeshCPU m;
    const float h = size * 0.5f;
    m.vertices = {
        {{-h, 0, -h}, {0, 1, 0}, {0, 0}},
        {{ h, 0, -h}, {0, 1, 0}, {1, 0}},
        {{ h, 0,  h}, {0, 1, 0}, {1, 1}},
        {{-h, 0,  h}, {0, 1, 0}, {0, 1}},
    };
    m.indices = {0, 1, 2, 0, 2, 3};
    return m;
}

MeshCPU uvSphere(uint32_t stacks, uint32_t slices, float radius) {
    if (stacks < 2 || slices < 3) {
        throw std::runtime_error("uvSphere: need stacks >= 2 and slices >= 3");
    }
    MeshCPU m;
    m.vertices.reserve((stacks + 1) * (slices + 1));
    m.indices.reserve(stacks * slices * 6);

    constexpr float kPi = 3.14159265358979323846f;
    for (uint32_t i = 0; i <= stacks; ++i) {
        const float v = (float)i / (float)stacks;
        const float phi = v * kPi;          // 0 .. pi
        const float sinP = std::sin(phi), cosP = std::cos(phi);
        for (uint32_t j = 0; j <= slices; ++j) {
            const float u = (float)j / (float)slices;
            const float theta = u * 2.f * kPi;
            const float sinT = std::sin(theta), cosT = std::cos(theta);
            MeshVertex mv;
            const float nx = sinP * cosT, ny = cosP, nz = sinP * sinT;
            mv.position[0] = radius * nx;
            mv.position[1] = radius * ny;
            mv.position[2] = radius * nz;
            mv.normal[0] = nx; mv.normal[1] = ny; mv.normal[2] = nz;
            mv.uv[0] = u; mv.uv[1] = v;
            m.vertices.push_back(mv);
        }
    }
    const uint32_t stride = slices + 1;
    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < slices; ++j) {
            const uint32_t a = i * stride + j;
            const uint32_t b = a + 1;
            const uint32_t c = a + stride;
            const uint32_t d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b,  b, c, d});
        }
    }
    return m;
}

MeshCPU torus(uint32_t ringSegments, uint32_t tubeSegments,
              float majorRadius, float minorRadius) {
    if (ringSegments < 3 || tubeSegments < 3) {
        throw std::runtime_error("torus: need ringSegments >= 3 and tubeSegments >= 3");
    }
    MeshCPU m;
    m.vertices.reserve((ringSegments + 1) * (tubeSegments + 1));
    m.indices.reserve(ringSegments * tubeSegments * 6);

    constexpr float kTwoPi = 6.28318530717958647692f;
    for (uint32_t i = 0; i <= ringSegments; ++i) {
        const float u = (float)i / (float)ringSegments;
        const float theta = u * kTwoPi;
        const float cosT = std::cos(theta), sinT = std::sin(theta);
        for (uint32_t j = 0; j <= tubeSegments; ++j) {
            const float v = (float)j / (float)tubeSegments;
            const float phi = v * kTwoPi;
            const float cosP = std::cos(phi), sinP = std::sin(phi);

            const float r = majorRadius + minorRadius * cosP;
            MeshVertex mv;
            mv.position[0] = r * cosT;
            mv.position[1] = minorRadius * sinP;
            mv.position[2] = r * sinT;
            mv.normal[0] = cosP * cosT;
            mv.normal[1] = sinP;
            mv.normal[2] = cosP * sinT;
            mv.uv[0] = u; mv.uv[1] = v;
            m.vertices.push_back(mv);
        }
    }
    const uint32_t stride = tubeSegments + 1;
    for (uint32_t i = 0; i < ringSegments; ++i) {
        for (uint32_t j = 0; j < tubeSegments; ++j) {
            const uint32_t a = i * stride + j;
            const uint32_t b = a + 1;
            const uint32_t c = a + stride;
            const uint32_t d = c + 1;
            m.indices.insert(m.indices.end(), {a, c, b,  b, c, d});
        }
    }
    return m;
}

}

}
