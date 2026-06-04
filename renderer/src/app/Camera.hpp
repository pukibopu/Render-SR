#pragma once

#include <simd/simd.h>

namespace rs {

class Camera {
public:
    Camera();

    void setPerspective(float fovY, float aspect, float near, float far);
    void setAspect(float aspect);
    void setOrbit(simd_float3 target, float azimuth, float elevation, float distance);

    void orbitAzimuth(float radians);
    void orbitElevation(float radians);
    void scaleDistance(float factor);

    simd_float4x4 view() const;
    simd_float4x4 proj() const;
    simd_float3   eye()  const;

private:
    simd_float3 m_target   {0.f, 0.f, 0.f};
    float       m_azimuth   = 0.0f;
    float       m_elevation = 0.35f;
    float       m_distance  = 4.0f;

    float m_fovY  = 1.0472f;   // 60 degrees
    float m_aspect = 16.f / 9.f;
    float m_near  = 0.1f;
    float m_far   = 100.f;
};

}
