#pragma once
#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "types.cuh"
#include "helper_math.cuh"

constexpr float EPSILON = 0.000001;




/*

__device__ __inline__ complex_float3 complex_cross_real(complex_float3 a, float3 b) {
    complex_float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__device__ __inline__ cuFloatComplex complex_dot_real(complex_float3 a, float3 b) {
    cuFloatComplex result;
    result = a.x * b.x + a.y * b.y + a.z * b.z;
    return result;
}*/

__device__ __inline__ complex_float3 complex_cross_real(const complex_float3 & a, const float3 & b) {
    complex_float3 result;
    result.x.x = __fmaf_rn(a.y.x, b.z,__fmaf_rn(-a.z.x, b.y, 0));
    result.x.y = __fmaf_rn(a.y.y, b.z,__fmaf_rn(-a.z.y, b.y, 0));

    result.y.x = __fmaf_rn(a.z.x, b.x, __fmaf_rn(-a.x.x, b.z, 0));
    result.y.y = __fmaf_rn(a.z.y, b.x, __fmaf_rn(-a.x.y, b.z, 0));

    result.z.x = __fmaf_rn(a.x.x, b.y, __fmaf_rn(-a.y.x, b.x, 0));
    result.z.y = __fmaf_rn(a.x.y, b.y, __fmaf_rn(-a.y.y, b.x, 0));

    return result;
}

__device__ __inline__ cuFloatComplex complex_dot_real(const complex_float3 & a, const float3 & b) {
    cuFloatComplex result;
    result.x = __fmaf_rn(a.x.x, b.x, __fmaf_rn(a.y.x, b.y, __fmaf_rn(a.z.x, b.z, 0)));
    result.y = __fmaf_rn(a.x.y, b.x, __fmaf_rn(a.y.y, b.y, __fmaf_rn(a.z.y, b.z, 0)));
    return result;

}




