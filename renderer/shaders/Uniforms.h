#pragma once

#include <simd/simd.h>

struct Uniforms {
    simd_float4x4 model;
    simd_float4x4 view;
    simd_float4x4 proj;
    simd_float3x3 normalMatrix;   // mat3(transpose(inverse(view * model)))
    simd_float3   lightDirView;   // direction the light travels, in view space
    float         ambient;
    simd_float3   albedo;
    float         specularStrength;
};
