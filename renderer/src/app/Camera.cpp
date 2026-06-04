#include "Camera.hpp"

#include <algorithm>
#include <cmath>

namespace rs {

namespace {

simd_float4x4 lookAtRH(simd_float3 eye, simd_float3 target, simd_float3 up) {
    const simd_float3 f = simd_normalize(target - eye);
    const simd_float3 s = simd_normalize(simd_cross(f, up));
    const simd_float3 u = simd_cross(s, f);

    simd_float4x4 m;
    m.columns[0] = simd_make_float4( s.x,  u.x, -f.x, 0.0f);
    m.columns[1] = simd_make_float4( s.y,  u.y, -f.y, 0.0f);
    m.columns[2] = simd_make_float4( s.z,  u.z, -f.z, 0.0f);
    m.columns[3] = simd_make_float4(-simd_dot(s, eye),
                                    -simd_dot(u, eye),
                                     simd_dot(f, eye), 1.0f);
    return m;
}

simd_float4x4 perspectiveRH_ZO(float fovY, float aspect, float n, float fr) {
    const float ys = 1.0f / std::tan(fovY * 0.5f);
    const float xs = ys / aspect;
    const float zs = fr / (n - fr);

    simd_float4x4 m;
    m.columns[0] = simd_make_float4(xs, 0.f, 0.f, 0.f);
    m.columns[1] = simd_make_float4(0.f, ys, 0.f, 0.f);
    m.columns[2] = simd_make_float4(0.f, 0.f, zs, -1.f);
    m.columns[3] = simd_make_float4(0.f, 0.f, n * zs, 0.f);
    return m;
}

}

Camera::Camera() = default;

void Camera::setPerspective(float fovY, float aspect, float near_, float far_) {
    m_fovY = fovY;
    m_aspect = aspect;
    m_near = near_;
    m_far = far_;
}

void Camera::setAspect(float aspect) { m_aspect = aspect; }

void Camera::setOrbit(simd_float3 target, float azimuth, float elevation, float distance) {
    m_target = target;
    m_azimuth = azimuth;
    m_elevation = elevation;
    m_distance = distance;
}

void Camera::orbitAzimuth(float radians) {
    m_azimuth += radians;
}

void Camera::orbitElevation(float radians) {
    constexpr float kLimit = 1.5533f;   // ~89 degrees
    m_elevation = std::clamp(m_elevation + radians, -kLimit, kLimit);
}

void Camera::scaleDistance(float factor) {
    m_distance = std::clamp(m_distance * factor, 0.25f, 100.0f);
}

simd_float3 Camera::eye() const {
    const float cosE = std::cos(m_elevation), sinE = std::sin(m_elevation);
    const float cosA = std::cos(m_azimuth),   sinA = std::sin(m_azimuth);
    return m_target + simd_make_float3(
        m_distance * cosE * sinA,
        m_distance * sinE,
        m_distance * cosE * cosA);
}

simd_float4x4 Camera::view() const {
    return lookAtRH(eye(), m_target, simd_make_float3(0.f, 1.f, 0.f));
}

simd_float4x4 Camera::proj() const {
    return perspectiveRH_ZO(m_fovY, m_aspect, m_near, m_far);
}

}
