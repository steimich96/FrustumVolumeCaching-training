
#pragma once

#include "helper_math.h"

namespace {
namespace device {   

struct Ray
{
    float3 origin;
    float3 dir;

    inline __device__ float3 at(float t) const { return origin + t * dir; }
};


inline __device__ float calculate_stepsize(const float t, const float cone_angle, const float dt_min, const float dt_max)
{
    return clamp(t * cone_angle, dt_min, dt_max);
}


inline __device__ __host__ int to1D(const int3 coords3D, const int3 dims)
{
    return coords3D.x * dims.y * dims.z + coords3D.y * dims.z + coords3D.z;
}
inline __device__ int to1DMulti(const int3 coords3D, const int3 dims, const int lvl, const int vals_per_lvl)
{
    const int lvl_offset = vals_per_lvl * lvl;
    return to1D(coords3D, dims) + lvl_offset;
}
inline __device__ int to1DMulti(const int3 coords3D, const int dims, const int lvl, const int vals_per_lvl)
{
    return to1DMulti(coords3D, make_int3(dims), lvl, vals_per_lvl);
}


inline __device__ float3 aabb_to_unit(const float3 point, const float3 aabb_from, const float3 aabb_to, const int grid_nlvl)
{
    return (point - aabb_from) / (aabb_to - aabb_from);
}

inline __device__ int mip_from_unit(const float3 point_unit)
{
    float3 scale = fabs(point_unit - 0.5f);
    float maxval = fmaxf(fmaxf(scale.x, scale.y), scale.z);

    // if maxval is almost zero, it will trigger frexpf to output 0 for exponent, which is not what we want.
    maxval = fmaxf(maxval, 0.1f);

    int exponent;
    frexpf(maxval, &exponent);
    int mip = max(0, exponent + 1);
    return mip;
}

template <int GRID_RESOLUTION>
inline __device__ bool grid_occupied_at_linear(const float3 point, const float3 aabb_from, const float3 aabb_to, const int grid_nlvl, const bool *grid_data)
{
    const float3 sample_point = aabb_to_unit(point, aabb_from, aabb_to, grid_nlvl);
    int mip = mip_from_unit(sample_point);

    if (mip >= grid_nlvl)
        return false;

    const float3 point_unit_in_mip = (sample_point - 0.5f) * scalbnf(1.0f, -mip) + 0.5f;
    const int3 idx3D_in_mip = make_int3(point_unit_in_mip * GRID_RESOLUTION);

    int idx = to1DMulti(idx3D_in_mip, GRID_RESOLUTION, mip, GRID_RESOLUTION * GRID_RESOLUTION * GRID_RESOLUTION);
    return grid_data[idx];
}


inline __device__ void _swap(float &a, float &b)
{
    float c = a;
    a = b;
    b = c;
}

inline __device__ bool ray_aabb_intersect(
    const Ray ray,
    const float3 aabb_from,
    const float3 aabb_to,
    float& near,
    float& far)
{
    // aabb is [xmin, ymin, zmin, xmax, ymax, zmax]
    float tmin = (aabb_from.x - ray.origin.x) / ray.dir.x;
    float tmax = (aabb_to.x -   ray.origin.x) / ray.dir.x;
    if (tmin > tmax)
        _swap(tmin, tmax);

    float tymin = (aabb_from.y - ray.origin.y) / ray.dir.y;
    float tymax = (aabb_to.y -   ray.origin.y) / ray.dir.y;
    if (tymin > tymax)
        _swap(tymin, tymax);

    if (tmin > tymax || tymin > tmax)
    {
        near = 1e10;
        far = 1e10;
        return false;
    }

    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (aabb_from.z - ray.origin.z) / ray.dir.z;
    float tzmax = (aabb_to.z   - ray.origin.z) / ray.dir.z;
    if (tzmin > tzmax)
        _swap(tzmin, tzmax);

    if (tmin > tzmax || tzmin > tmax)
    {
        near = 1e10;
        far = 1e10;
        return false;
    }

    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    near = tmin;
    far = tmax;
    return true;
}

} // namespace device
} // namespace