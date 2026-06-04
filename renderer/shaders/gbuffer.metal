#include <metal_stdlib>
#include "Uniforms.h"

using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
    float2 uv       [[attribute(2)]];
};

struct VertexOut {
    float4 position   [[position]];
    float3 normalView;
    float3 posView;
};

struct GBufferOut {
    float4 color  [[color(0)]];   // RGBA8Unorm_sRGB
    float  depth  [[color(1)]];   // R32Float,    linear eye-space depth
    float4 normal [[color(2)]];   // RGBA16Float, view-space normal in [-1,1]
};

vertex VertexOut gbuffer_vertex(VertexIn in [[stage_in]],
                                constant Uniforms& u [[buffer(1)]])
{
    float4 worldPos = u.model * float4(in.position, 1.0);
    float4 viewPos  = u.view  * worldPos;
    VertexOut out;
    out.position   = u.proj * viewPos;
    out.posView    = viewPos.xyz;
    out.normalView = u.normalMatrix * in.normal;
    return out;
}

static float3 shade(float3 N, float3 posView, constant Uniforms& u)
{
    float3 L = normalize(-u.lightDirView);   // toward light source
    float3 V = normalize(-posView);          // toward camera (eye at origin in view space)
    float3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = (diff > 0.0) ? pow(max(dot(N, H), 0.0), 32.0) : 0.0;

    return u.albedo * (u.ambient + diff) + float3(u.specularStrength) * spec;
}

fragment GBufferOut gbuffer_fragment(VertexOut in [[stage_in]],
                                     constant Uniforms& u [[buffer(1)]])
{
    float3 N = normalize(in.normalView);

    GBufferOut out;
    out.color  = float4(shade(N, in.posView, u), 1.0);
    out.depth  = -in.posView.z;              // linear eye-space, positive into scene
    out.normal = float4(N, 0.0);
    return out;
}

fragment float4 rgb_fragment(VertexOut in [[stage_in]],
                             constant Uniforms& u [[buffer(1)]])
{
    float3 N = normalize(in.normalView);
    return float4(shade(N, in.posView, u), 1.0);
}
