#pragma once
#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"

#include "vector_types.h"

#include <iostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "cuComplex.h"
#include <math.h>
#include "math_constants.h"

#include "gpu_error.cuh"

__device__ __inline__ complex_float3 ray_launch(const complex_float3 & e_field,float3 ray){
    //vectors that are the 3 axis of cartesian space
    float3 x_axis = make_float3(1,0,0);
    float3 y_axis = make_float3(0,1,0);
    float3 z_axis = make_float3(0,0,1);

    float x_norm = length(cross(x_axis,ray));
    float y_norm = length(cross(y_axis,ray));
    float z_norm = length(cross(z_axis,ray));


    float3 ray_u;
    if (abs(x_norm) > abs(y_norm) && abs(x_norm) > abs(z_norm)){
        ray_u = normalize(cross(x_axis,ray));
    }
    else if (abs(y_norm) >= abs(x_norm) && abs(y_norm) > abs(z_norm)){
        ray_u = normalize(cross(y_axis,ray));
    }
    else if (abs(z_norm) >= abs(x_norm) && abs(z_norm) >= abs(y_norm)){
        ray_u = normalize(cross(z_axis,ray));
    }
    float3 ray_v = normalize(cross(ray,ray_u));


    cuFloatComplex e_field_axis_u = complex_dot_real(e_field,ray_u);
    cuFloatComplex e_field_axis_v  = complex_dot_real(e_field,ray_v);
    
    complex_float3 ray_field;
    /*
    ray_field.x =e_field_axis_u * ray_u.x + e_field_axis_v * ray_v.x;
    ray_field.y =e_field_axis_u * ray_u.y + e_field_axis_v * ray_v.y;
    ray_field.z =e_field_axis_u * ray_u.z + e_field_axis_v * ray_v.z;
    */
    //change to explicit __fmaf_rn
    ray_field.x.x = __fmaf_rn(e_field_axis_u.x, ray_u.x, __fmaf_rn(e_field_axis_v.x, ray_v.x, 0));
    ray_field.x.y = __fmaf_rn(e_field_axis_u.y, ray_u.x, __fmaf_rn(e_field_axis_v.y, ray_v.x, 0));
    ray_field.y.x = __fmaf_rn(e_field_axis_u.x, ray_u.y, __fmaf_rn(e_field_axis_v.x, ray_v.y, 0));
    ray_field.y.y = __fmaf_rn(e_field_axis_u.y, ray_u.y, __fmaf_rn(e_field_axis_v.y, ray_v.y, 0));
    ray_field.z.x = __fmaf_rn(e_field_axis_u.x, ray_u.z, __fmaf_rn(e_field_axis_v.x, ray_v.z, 0));
    ray_field.z.y = __fmaf_rn(e_field_axis_u.y, ray_u.z, __fmaf_rn(e_field_axis_v.y, ray_v.z, 0));
    

    return ray_field;
}


__device__ __inline__ complex_float3 em_wave(const double2 alpha_beta, const float4& ray, const PointData& origin, const PointData& end, float wave_length) {
    //printf(("package- alpha_beta = (%f, %f)\n", alpha_beta.x, alpha_beta.y);
    printf("package- ray field %f +i%f, %f +i%f, %f +i%f\n", origin.electric_field.x.x, origin.electric_field.x.y, origin.electric_field.y.x, origin.electric_field.y.y, origin.electric_field.z.x, origin.electric_field.z.y);



    complex_float3 ray_field = ray_launch(origin.electric_field, make_float3(ray.x, ray.y, ray.z));
    printf("package- ray_field after launch = (%f + %fi, %f + %fi, %f + %fi)\n", ray_field.x.x, ray_field.x.y, ray_field.y.x, ray_field.y.y, ray_field.z.x, ray_field.z.y);


    float front = -(1 / (2 * CUDART_PI_F));

    cuFloatComplex G;
   // printf("inputs to sincosf = (%f)\n", -alpha_beta.y * ray.w);
   // printf("alpha  = (%f)\n", alpha_beta.x);
    printf("alpha, beta  = (%f,  %f)\n", alpha_beta.x, alpha_beta.y);
    sincosf(-alpha_beta.y * ray.w, &G.y, &G.x);
   // printf("dis2t %f \n", ray.w);
   // printf("package- G after sincosf = (%f + %fi)\n", G.x, G.y);

    G *= (expf(-alpha_beta.x * ray.w)  / ray.w);
   // printf("package- G after exp/log = (%f + %fi)\n", G.x, G.y);

    cuFloatComplex dG;
    dG.x = -alpha_beta.x - (1 / ray.w);
    dG.y = -alpha_beta.y;
    //printf(("package- dG before mul = (%f + %fi)\n", dG.x, dG.y);
    printf("dg pre mul with g = (%f + %fi)\n", dG.x, dG.y);

    dG = cuCmulf(dG, G);
   // printf("package- dG after mul = (%f + %fi)\n", dG.x, dG.y);
    //printf(("package- front = %f\n", front);
    //printf(("package- dG = (%f + %fi)\n", dG.x, dG.y);
   // printf("package- end.normal = (%f, %f, %f)\n", end.normal.x, end.normal.y, end.normal.z);

    float3 ray_dir = make_float3(ray.x, ray.y, ray.z);
   // printf("package- ray direction = (%f, %f, %f)\n", ray_dir.x, ray_dir.y, ray_dir.z);

    float dot_val = dot(end.normal, ray_dir);
    printf("front = %f\n", front);

   printf(" dot(end.normal, ray_dir) = %f\n", dot_val);
   printf("dg = (%f + %fi)\n", dG.x, dG.y);
    printf("G = (%f + %fi)\n", G.x, G.y);
    printf("ray_dir = (%f, %f, %f)\n", ray_dir.x, ray_dir.y, ray_dir.z);
    printf("raylength = %f\n", ray.w);
    cuFloatComplex loss = front * dG * dot(end.normal, make_float3(ray.x, ray.y, ray.z));
   printf("package- loss = (%f + %fi)\n", loss.x, loss.y);

    ray_field *= loss;


    printf("ray_field after loss = (%f + %fi, %f + %fi, %f + %fi)\n", ray_field.x.x, ray_field.x.y, ray_field.y.x, ray_field.y.y, ray_field.z.x, ray_field.z.y);


    return ray_field;
}







