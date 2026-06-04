#include <metal_stdlib>
using namespace metal;

struct Vertex {
    packed_float2 position;
    packed_float3 color;
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
};

vertex VertexOut triangle_vertex(uint vid [[vertex_id]],
                                 device const Vertex* verts [[buffer(0)]]) {
    VertexOut out;
    out.position = float4(verts[vid].position, 0.0, 1.0);
    out.color    = verts[vid].color;
    return out;
}

fragment float4 triangle_fragment(VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
