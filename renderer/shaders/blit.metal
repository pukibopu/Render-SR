#include <metal_stdlib>

using namespace metal;

struct BlitVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen-triangle trick: 3 vertices, no vertex buffer needed.
// Covers the [-1,1] clip-space square with one triangle whose UVs
// interpolate to [0,1] inside the visible area.
vertex BlitVtxOut blit_vertex(uint vid [[vertex_id]])
{
    const float2 positions[3] = {
        float2(-1.0, -3.0),
        float2(-1.0,  1.0),
        float2( 3.0,  1.0),
    };
    const float2 uvs[3] = {
        float2(0.0, 2.0),
        float2(0.0, 0.0),
        float2(2.0, 0.0),
    };
    BlitVtxOut out;
    out.position = float4(positions[vid], 0.0, 1.0);
    out.uv       = uvs[vid];
    return out;
}

fragment float4 blit_rgb_fragment(BlitVtxOut in [[stage_in]],
                                  texture2d<float> src [[texture(0)]],
                                  sampler s            [[sampler(0)]])
{
    return float4(src.sample(s, in.uv).rgb, 1.0);
}

// depthRange.x = near (mapped to 0), depthRange.y = far (mapped to 1).
fragment float4 blit_depth_fragment(BlitVtxOut in [[stage_in]],
                                    texture2d<float> src         [[texture(0)]],
                                    sampler s                    [[sampler(0)]],
                                    constant float2& depthRange  [[buffer(0)]])
{
    float d = src.sample(s, in.uv).r;
    float t = saturate((d - depthRange.x) / max(depthRange.y - depthRange.x, 1e-6));
    return float4(t, t, t, 1.0);
}

fragment float4 blit_normal_fragment(BlitVtxOut in [[stage_in]],
                                     texture2d<float> src [[texture(0)]],
                                     sampler s            [[sampler(0)]])
{
    float3 n = src.sample(s, in.uv).xyz;
    return float4(n * 0.5 + 0.5, 1.0);
}
