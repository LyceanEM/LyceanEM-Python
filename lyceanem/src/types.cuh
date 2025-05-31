#pragma once
#include "cuComplex.h"
#include <math.h>



struct __host__ __device__ complex_float3 {
    cuFloatComplex x;
    cuFloatComplex y;
    cuFloatComplex z;


    // Example function to add two complex_float3 instances
    __host__ __device__ __inline__  void  operator/=( const float& b) {
        x.x/=b;
        x.y/=b;
        y.x/=b;
        y.y/=b;
        z.x/=b;
        z.y/=b;
    }
    __host__ __device__  __inline__  void  operator+=( const complex_float3& b) {
        x.x += b.x.x;
        x.y += b.x.y;
        y.x += b.y.x;
        y.y += b.y.y;
        z.x += b.z.x;
        z.y += b.z.y;
    }
    __host__ __device__  __inline__  void  operator=( const complex_float3& b) {
        x.x = b.x.x;
        x.y = b.x.y;
        y.x = b.y.x;
        y.y = b.y.y;
        z.x = b.z.x;
        z.y = b.z.y;
    }

    __host__ __device__  __inline__  void  operator*=(  const complex_float3& b) {
        complex_float3 result;
        result.x.x = x.x * b.x.x - x.y * b.x.y;
        result.x.y = x.x * b.x.y + x.y * b.x.x;
        result.y.x = y.x * b.y.x - y.y * b.y.y;
        result.y.y = y.x * b.y.y + y.y * b.y.x;
        result.z.x = z.x * b.z.x - z.y * b.z.y;
        result.z.y = z.x * b.z.y + z.y * b.z.x;
        x = result.x;
        y = result.y;
        z = result.z;

    }
    __host__ __device__  __inline__  void  operator*=(  const cuFloatComplex& b) {
        complex_float3 result;

        result.x.x = x.x * b.x - x.y * b.y;
        result.x.y = x.x * b.y + x.y * b.x;
        result.y.x = y.x * b.x - y.y * b.y;
        result.y.y = y.x * b.y + y.y * b.x;
        result.z.x = z.x * b.x - z.y * b.y;
        result.z.y = z.x * b.y + z.y * b.x;
        x = result.x;
        y = result.y;
        z = result.z;

    }
    __host__ __device__  __inline__  complex_float3  operator*( const float& b) {
        complex_float3 result;
        result.x.x = x.x * b;
        result.x.y = x.y * b;
        result.y.x = y.x * b;
        result.y.y = y.y * b;
        result.z.x = z.x * b;
        result.z.y = z.y * b;

        return result;


    }


      __device__  __inline__  void  plusEqualAtomic( const    complex_float3& b) {
        float * xreal = (float*)&x.x;
        float * ximag = (float*)&x.y;
        float * yreal = (float*)&y.x;
        float * yimag = (float*)&y.y;
        float * zreal = (float*)&z.x;
        float * zimag = (float*)&z.y;
        atomicAdd(xreal,  b.x.x);
        atomicAdd(ximag,  b.x.y);
        atomicAdd(yreal,  b.y.x);
        atomicAdd(yimag,  b.y.y);
        atomicAdd(zreal,  b.z.x);
        atomicAdd(zimag,  b.z.y);
    }



};
// point data struct
struct __host__ __device__ PointData {
    complex_float3 electric_field;
    float3 normal;

};


PointData __host__ __device__  create_point_data(float ex_real, float ex_imag, float ey_real, float ey_imag, float ez_real, float ez_imag, float normal_x, float normal_y, float normal_z) {
PointData point_data;
point_data.electric_field.x = make_cuFloatComplex(ex_real, ex_imag);
point_data.electric_field.y = make_cuFloatComplex(ey_real, ey_imag);
point_data.electric_field.z = make_cuFloatComplex(ez_real, ez_imag);
point_data.normal = make_float3(normal_x, normal_y, normal_z);

return point_data;
}