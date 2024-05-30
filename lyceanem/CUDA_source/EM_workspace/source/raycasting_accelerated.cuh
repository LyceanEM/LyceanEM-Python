#pragma once

#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"
#include "gpu_error.cuh"


#include <iostream>
#include <pybind11/numpy.h>
#include <assert.h>
#include <math.h> // for math functions like floorf

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "terrain_acceleration_build.cuh"
__device__ __inline__ float3 ray_plane_intersection(float3 ray_direction, float3 ray_origin, float3 plane_normal, float3 plane_origin){
    float  d = dot(ray_origin - plane_origin, plane_normal);
    float t = dot(plane_normal, ray_direction);
    float3 point = ray_origin - d / t * ray_direction;

    // check on line
    return point;

}
__device__ __inline__ int2 bin_point_2d( float2 y_range, float2 z_range, float3 point){

    int biny = floorf((point.y-y_range.x)/y_range.y);
    int binz = floorf((point.z-z_range.x)/z_range.y);
    ////printf("BINNING %f %f %f  %i\n",point.y,y_range.x,y_range.y,biny);


    return make_int2(biny, binz);
}










__device__ __inline__ float2 gradient_component_y_z_2d(float4 ray_direction){
    float deltaDisty = (ray_direction.y == 0) ? 1e30 : std::abs(1/ ray_direction.y);
    float deltaDistz = (ray_direction.z == 0) ? 1e30 : std::abs(1 / ray_direction.z);

    return make_float2( deltaDisty, deltaDistz);
}
__device__ __inline__ int2 step_direction(float4 ray_direction){
    int stepy = (ray_direction.y < 0) ? -1 : 1;
    int stepz = (ray_direction.z < 0) ? -1 : 1;
    return make_int2(stepy, stepz);
}







__device__ __inline__ bool intersection2(int i, float4 *ray, float3 *tri_vertex,int3* triangle_index, int tri_num, float3 origin,int end_num,int flag)
{
    


        //rays are stored in end order major
        int c = i / end_num;
        int d = i % end_num;

       
        int result = 0;
        ////printf("hi\n");
        if(flag || c!= d)
        {
            int b;
            float4 r = ray[i];
            ////printf("hi post ray access %i\n", tri_num);


            for(int t = 0 ; t < tri_num ; t++)
            {

                b = hit(
                    make_float3(r.x,r.y,r.z),
                    r.w,
                    origin,
                    tri_vertex[triangle_index[t].y] - tri_vertex[triangle_index[t].x],
                    tri_vertex[triangle_index[t].z] -tri_vertex[triangle_index[t].x],
                    tri_vertex[triangle_index[t].x]
                );
             
                result = (b == 1) ? 1 : result;


                   
            }


        }

     
        return result;
 


    
}


__device__ __inline__ void DDA(float3* source, float3 * end, float4 *rays, int2* ray_point_index, int ray_index,
    int3 *binned_triangles, float3 *tri_vertex, int2* tri_num_in_bin,
     int2 num_bins, int end_num, int flag, int x_offset, int y_offset, float2 x_top_bottom, float2 y_range, float2 z_range)
{
    int store = ray_index;
    float4 ray_direction = rays[ray_index];
    int c = ray_index / end_num;
    int d = ray_index % end_num;

    float3 ray_origin = source[c];
    float3 endss = end[d];



    // find origin and end tiles

    float3 ray_dir = make_float3(ray_direction.x,ray_direction.y,ray_direction.z);
    float3 top_intersection = ray_plane_intersection(ray_dir, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.y,y_range.x,z_range.x));
    float3 bottom_intersection = ray_plane_intersection(ray_dir, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.x,y_range.x,z_range.x));
    const int2 origional_tile_origin = bin_point_2d( y_range, z_range, top_intersection);
    const int2 tile_destination = bin_point_2d( y_range, z_range, bottom_intersection);

    // gradient of ray component wise
    float2 gradient = gradient_component_y_z_2d(ray_direction);

    ////printf("gradient %f %f %f %f \n",gradient.x,gradient.y,endss.y-ray_origin.y, endss.z-ray_origin.z);
    //what direction to step in x or y-direction (either +1 or -1)
    int2 step = step_direction(ray_direction);
    double2 distance_to_next_side;
    //check step lines ip with box order

    // temporary fix for when the ray is coming from bellow-------------------------------------
    if(step.x == 1){
        if(bottom_intersection.y< top_intersection.y){
            step.x = -1;
        }
    }
    if(step.y == 1){
        if(bottom_intersection.z< top_intersection.z){
            step.y = -1;
        }
    }
    if(step.x == -1){
        if(bottom_intersection.y> top_intersection.y){
            step.x = 1;      
        }
    }
    if(step.y == -1){
        if(bottom_intersection.z> top_intersection.z){
            step.y = 1;      
        }
    }
    float2 top_intersection_in_fractional_bin_units = make_float2((top_intersection.y-y_range.x)/y_range.y,(top_intersection.z-z_range.x)/z_range.y);
    int2 tile_origin = make_int2(origional_tile_origin.x,origional_tile_origin.y);

   


    
    if(step.x == 1){
        distance_to_next_side.x = ((tile_origin.x + 1.0) - top_intersection_in_fractional_bin_units.x) * gradient.x;
    }
    else{
        distance_to_next_side.x = (top_intersection_in_fractional_bin_units.x - tile_origin.x) * gradient.x;
    }
    if(step.y == 1){
        distance_to_next_side.y = ((tile_origin.y + 1.0) - top_intersection_in_fractional_bin_units.y) * gradient.y;
    }
    else{
        distance_to_next_side.y = (top_intersection_in_fractional_bin_units.y - tile_origin.y) * gradient.y;
    }
    assert(distance_to_next_side.x >= 0);
    assert(distance_to_next_side.y >= 0);

    bool arrived_at_destination = false;
    bool intersection_with_triangle = false;
    bool intersected = false;
    bool outside_map = (tile_origin.x < 0 || tile_origin.x >= num_bins.x || tile_origin.y < 0 || tile_origin.y >= num_bins.y);
    gradient.x/=2;
    gradient.y/=2;
////////////////////////////////////////////////////////////////////////////////cell_index is for bin_num
/////////////////////////////////////////////////////////////////////////////// tri_index needs accumulated bin_num
int x_ranfe = std::abs(tile_destination.x-tile_origin.x)+1;
int y_ranfe = std::abs(tile_destination.y-tile_origin.y)+1;


for( int i = 0; i < x_ranfe; i++){
        for( int j = 0; j < y_ranfe; j++){
           // printf("I %i J %i\n",i*step.x+tile_origin.x,j*step.y+tile_origin.y);

            int cell_index = (i*step.x+tile_origin.x)* num_bins.y + j*step.y + tile_origin.y;
            outside_map = (i < 0 || i >= num_bins.x || j < 0 || j >= num_bins.y);
            if(!outside_map){
                intersection_with_triangle = intersection2(ray_index,
                                                        rays,
                                                        tri_vertex, 
                                                        (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                        tri_num_in_bin[cell_index].x, 
                                                        ray_origin, 
                                                            end_num, flag);

            }
            arrived_at_destination = (i+tile_origin.x == tile_destination.x && j+tile_origin.y == tile_destination.y);
            if(arrived_at_destination){
            }
            if(intersection_with_triangle){
                i = x_ranfe;
                j = y_ranfe;
                intersected = true;
            
            }
    }
    }
    if(intersected){
        ray_point_index[ray_index] = make_int2(-1,-1);
    }
    else{
        ray_point_index[ray_index] = make_int2(c+x_offset,d+y_offset);
    }

/*
    ////printf("cell_index %i\n",cell_index);
    
    int cell_index = tile_origin.x * num_bins.y + tile_origin.y;
    if(!outside_map){
        intersection_with_triangle = intersection2(ray_index,
                                                rays,
                                                tri_vertex, 
                                                (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                tri_num_in_bin[cell_index].x, 
                                                ray_origin, end_num, flag);
    }

    



    arrived_at_destination = (tile_origin.x == tile_destination.x && tile_origin.y == tile_destination.y);

    bool x_done = (tile_origin.x==tile_destination.x);
    bool y_done = (tile_origin.y==tile_destination.y); 
    
    
    ////printf("tile_origin %i %i, tile_destination %i %i, step %i %i gradient %f,%f\n",tile_origin.x,tile_origin.y,tile_destination.x,tile_destination.y,step.x,step.y,gradient.x,gradient.y);
    ////printf("inputs tp side distance %i %f %f\n",tile_origin.x, top_intersection.x, y_range.x);
    ////printf("inputs tp side distance %i %f %f\n",tile_origin.y, top_intersection.y, z_range.x);
    ////printf("distance_to_next_side %f %f\n",distance_to_next_side.x,distance_to_next_side.y);
    float epsilon;
    if(gradient.x > gradient.y){
        epsilon = gradient.x/5;
    }
    else{
        epsilon = gradient.y/5;
    }

    int jjj = 0;
    //printf("numbins %i %i\n",num_bins.x,num_bins.y);

    
    while (!arrived_at_destination && !intersection_with_triangle){
        jjj++;
        //jump to next map square, OR in x-direction, OR in y-direction
        x_done = (tile_origin.x==tile_destination.x);
        y_done = (tile_origin.y==tile_destination.y); 
        float difference_distance = distance_to_next_side.x - distance_to_next_side.y;
        if (distance_to_next_side.x < distance_to_next_side.y){
            distance_to_next_side.x += gradient.x;
            distance_to_next_side.y -= gradient.x;
            tile_origin.x += step.x;
            arrived_at_destination = (tile_origin.x == tile_destination.x && tile_origin.y == tile_destination.y);
            x_done = (tile_origin.x==tile_destination.x);
            if(std::abs(difference_distance) < epsilon){
                
                int cell_index = tile_origin.x * num_bins.y + tile_origin.y+step.y;

                ////printf("cell_index bin num %i %i,%i\n",cell_index,tile_origin.x,tile_origin.y);
                ////printf("map_dimens %i %i\n",num_bins.x,num_bins.y);
                outside_map = (tile_origin.x < 0 || tile_origin.x >= num_bins.x || tile_origin.y+step.y < 0 || tile_origin.y+step.y >= num_bins.y);
                if(!outside_map && !arrived_at_destination && !intersection_with_triangle){
                    ////printf("inside map\n");
                    intersection_with_triangle = intersection2(ray_index,
                                                            rays,
                                                            tri_vertex, 
                                                            (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                            tri_num_in_bin[cell_index].x, 
                                                            ray_origin, 
                                                                end_num, flag);
                }
                arrived_at_destination = (arrived_at_destination || (tile_origin.x == tile_destination.x && tile_origin.y + step.y == tile_destination.y));
                y_done = (tile_origin.y+step.y==tile_destination.y);

            }
        }

        //else if(distance_to_next_side.x > distance_to_next_side.y+epsilon.y){
        else{
            distance_to_next_side.y += gradient.y;
            distance_to_next_side.x -= gradient.y;
            tile_origin.y += step.y;
            y_done = (tile_origin.y==tile_destination.y);
            arrived_at_destination = (tile_origin.x == tile_destination.x && tile_origin.y == tile_destination.y);
            if(std::abs(difference_distance) < epsilon){
                int cell_index = (tile_origin.x+step.x) * num_bins.y + tile_origin.y;

                ////printf("cell_index bin num %i %i,%i\n",cell_index,tile_origin.x,tile_origin.y);
                ////printf("map_dimens %i %i\n",num_bins.x,num_bins.y);
                outside_map = (tile_origin.x+step.x < 0 || tile_origin.x+step.x >= num_bins.x || tile_origin.y < 0 || tile_origin.y >= num_bins.y);
                if(!outside_map && !arrived_at_destination && !intersection_with_triangle){
                    ////printf("inside map\n");
                    intersection_with_triangle = intersection2(ray_index,
                                                            rays,
                                                            tri_vertex, 
                                                            (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                            tri_num_in_bin[cell_index].x, 
                                                            ray_origin, 
                                                                end_num, flag);
                }
                arrived_at_destination = arrived_at_destination = (arrived_at_destination || (tile_origin.x + step.x == tile_destination.x && tile_origin.y == tile_destination.y));
                x_done = (tile_origin.x+step.x==tile_destination.x);   
            }
        }
 
        
           // assert(shouldnthappen);
        
 


        ////printf("tile_curret %i, %i destination %i %i\n",tile_origin.x,tile_origin.y,tile_destination.x,tile_destination.y);
        int cell_index = tile_origin.x * num_bins.y + tile_origin.y;

        ////printf("cell_index bin num %i %i,%i\n",cell_index,tile_origin.x,tile_origin.y);
        ////printf("map_dimens %i %i\n",num_bins.x,num_bins.y);
        outside_map = (tile_origin.x < 0 || tile_origin.x >= num_bins.x || tile_origin.y < 0 || tile_origin.y >= num_bins.y);
        if(!outside_map && !arrived_at_destination && !intersection_with_triangle){
            ////printf("inside map\n");
            intersection_with_triangle = intersection2(ray_index,
                                                    rays,
                                                    tri_vertex, 
                                                    (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                    tri_num_in_bin[cell_index].x, 
                                                    ray_origin, 
                                                        end_num, flag);
        } 


        ////printf("tile_current %i,%i, destination %i,%i\n",tile_origin.x,tile_origin.y,tile_destination.x,tile_destination.y);
        bool run_away_too_far_x = ((tile_origin.x> tile_destination.x && step.x ==1) || (tile_origin.x < tile_destination.x && step.x == -1 ));
        bool run_away_too_far_y = ((tile_origin.y> tile_destination.y && step.y ==1 ) || (tile_origin.y < tile_destination.y && step.y == -1 ));
        assert(!run_away_too_far_x);
        assert(!run_away_too_far_y );

        
    }
    if(intersection_with_triangle){
        ray_point_index[ray_index] = make_int2(-1,-1);
    }
    else if(!arrived_at_destination && ((x_done && !y_done) || (y_done && !x_done))){
        if(x_done){
            for(int i = tile_origin.y; i != tile_destination.y+step.y; i+=step.y){
                int cell_index = tile_origin.x * num_bins.y + tile_origin.y;
                outside_map = (tile_origin.x < 0 || tile_origin.x >= num_bins.x || tile_origin.y < 0 || tile_origin.y >= num_bins.y);
                if(!outside_map){
                    intersection_with_triangle = intersection2(ray_index,
                                                            rays,
                                                            tri_vertex, 
                                                            (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                            tri_num_in_bin[cell_index].x, 
                                                            ray_origin, 
                                                                end_num, flag);
                }
            }
        }
        else{
            for(int i = tile_origin.x; i != tile_destination.x+step.x; i+=step.x){
                int cell_index = tile_origin.x * num_bins.y + tile_origin.y;
                outside_map = (tile_origin.x < 0 || tile_origin.x >= num_bins.x || tile_origin.y < 0 || tile_origin.y >= num_bins.y);
                if(!outside_map){
                    intersection_with_triangle = intersection2(ray_index,
                                                            rays,
                                                            tri_vertex, 
                                                            (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                            tri_num_in_bin[cell_index].x, 
                                                            ray_origin, 
                                                                end_num, flag);
                }

        
            }
          
        }
        if(intersection_with_triangle){
            //printf("intersection_with_triangle\n");
            ray_point_index[ray_index] = make_int2(-1,-1);

        }
        else{
            ray_point_index[ray_index] = make_int2(c+x_offset,d+y_offset);
        }
    
    }
    else
    {
        ray_point_index[ray_index] = make_int2(c+x_offset,d+y_offset);
    }
    
*/
}


__device__ __inline__ complex_float3 ray_launch2(const complex_float3 & e_field,float3 ray){
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


__device__ __inline__ complex_float3 em_wave2(int ray_wave, const float4& ray, const PointData& origin, const PointData& end,float wave_length, float alpha, float beta) {


    complex_float3 ray_field;
    if (ray_wave == 0){

        if (origin.is_electric){
            ray_field = origin.electric_field;

        }
        else{
            float source_impedance = sqrt(origin.permeability.x / origin.permittivity.x);
            //printf("source_impedance %f\n",source_impedance);
            ray_field =  complex_cross_real(origin.electric_field, make_float3(ray.x,ray.y,ray.z)); 
            ray_field /= source_impedance;

        }

    }


    
    
    complex_float3 ray_field2 = ray_launch2(ray_field,make_float3(ray.x,ray.y,ray.z));



    cuFloatComplex loss;
    float spherical_expansion_loss = (wave_length / (4 * CUDART_PI_F * ray.w));

    cuFloatComplex loss_phase;
    sincosf(-ray.w*beta, &loss_phase.y, &loss_phase.x);
    cuFloatComplex loss_medium;
    sincosf(-ray.w*alpha, &loss_medium.y, &loss_medium.x);

    loss = loss_medium * loss_phase * spherical_expansion_loss;

    // old loss


    ray_field2 *= loss;



    return ray_field2;

}





__global__ void raycast2(float3 *source, float3 *end, float4 *ray, int source_num, int end_num, int flag, float3 *tri_vertex,int3 *binned_triangles, int2* tri_num_in_bin, int ray_num,
     int2 *ray_index, int x_offset, int y_offset, int2 num_bins, float2 x_top_bottom, float2 y_range, float2 z_range,PointData* points, float wave_length, complex_float3* scattering_network, float alpha, float beta)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = thread; i < ray_num; i+=stride)
    {

        points_to_rays(i,source,end,ray,source_num,end_num,flag);

        DDA(source, end, ray,ray_index,i,binned_triangles,tri_vertex,tri_num_in_bin,num_bins,end_num,flag,x_offset,y_offset,x_top_bottom,y_range,z_range);
        if(ray_index[i].x != -1){
            complex_float3 ray_field = em_wave2(0,ray[i],points[ray_index[i].x],points[ray_index[i].y],wave_length,alpha,beta);
            scattering_network[i] = ray_field;
        }


    }

}

__global__ void set_values(int2 *ray_index, int ray_num, int2 value)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i = thread; i < ray_num; i+=stride)
    {
        ray_index[i] = value;
    }
}
__global__ void printer(int2 *ray_index, int ray_num)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i = thread; i < ray_num; i+=stride)
    {
        printf("ray_index %i %i\n",ray_index[i].x,ray_index[i].y);
    }
}




void raycast_wrapper2 (float *source, float *end, int source_num, int end_num, float3 *d_tri_vertex,int3 *d_binned_triangles, int2* d_tri_num_per_bin, int2 num_bins, float2 x_top_bottom, float2 y_range, float2 z_range,
    PointData* points, float wave_length, complex_float3* h_scattering_network,float alpha,float beta)
{
    // declare device memory
    float3 *d_source;
    float3 *d_end;

    float4 *d_ray;
    int2 *d_ray_index;



    // calculate size of arrays
    int source_size = source_num * sizeof(float3);
    int end_size = end_num * sizeof(float3);

    int ray_size = (source_num * end_num+ source_num ) * sizeof(float4);
    int ray_index_size = (source_num * end_num+ source_num ) * sizeof(int2);


    // allocate memory on device
    cudaMalloc((void**)&d_source, source_size);
    size_t total_free = 0;
    size_t free, total;
    cudaSetDevice(0);
    cudaMemGetInfo(&free, &total);
    free *= 0.95;
    total_free += free;
    
    cudaMalloc((void**)&d_end, end_size);

    
    cudaMalloc((void**)&d_ray, ray_size);
    cudaMalloc((void**)&d_ray_index, ray_index_size);
    gpuErrchk( cudaGetLastError() );




    // copy data to device
    cudaMemcpy(d_source,(float3*) source, source_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_end,(float3*) end, end_size, cudaMemcpyHostToDevice);


    
    gpuErrchk( cudaGetLastError() );

    
    // set data to zero in 
    float3 zero = make_float3(0,0,0);

    cudaMemset(d_ray, 0, ray_size);
    set_values<<<32,256>>>(d_ray_index, (source_num * end_num),make_int2(-1,-1));
    gpuErrchk( cudaGetLastError() );


 
    //launch kernel for each ray_wave

    // source to sink
    bool not_self_to_self = (d_source != d_end);
    PointData* d_points;
    complex_float3* d_scattering_network;

    // calculate size of arrays
    int points_size = (source_num+end_num) * sizeof(PointData);

    int scattering_network_size = (end_num*source_num) * sizeof(complex_float3);
    auto start = std::chrono::high_resolution_clock::now();

    
    // allocate device memory
    cudaMalloc((void**)&d_points, points_size);
    cudaMalloc((void**)&d_scattering_network, scattering_network_size);
    gpuErrchk( cudaGetLastError() );

    // copy data to device
    cudaMemcpy(d_points, points, points_size, cudaMemcpyHostToDevice);
    gpuErrchk( cudaGetLastError() );



    // set to 0
    cudaMemset(d_scattering_network, 0, scattering_network_size);
    cudaDeviceSynchronize();

    raycast2<<<32,256>>>(d_source,d_end,d_ray,source_num,end_num,not_self_to_self,d_tri_vertex,d_binned_triangles,d_tri_num_per_bin,source_num*end_num, d_ray_index,0,source_num,num_bins,x_top_bottom,y_range,z_range,d_points,wave_length,d_scattering_network,alpha,beta);
    //get last error
    gpuErrchk( cudaGetLastError() );

    //printer<<<32,128>>>(d_ray_index,source_num*end_num);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    /*
    gpuErrchk( cudaGetLastError() );
    int rays_filled = source_num * end_num;

    // source to scatter
        not_self_to_self = (d_source != d_scatter);
        raycast2<<<32,128>>>(d_source,d_scatter,&d_ray[rays_filled], source_num, scatter_num, not_self_to_self,d_tri_vertex,d_binned_triangles,d_tri_num_per_bin,source_num*scatter_num, &d_ray_index[rays_filled],0,source_num+end_num,num_bins,x_top_bottom,y_range,z_range);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        gpuErrchk( cudaGetLastError() );
        rays_filled += source_num * scatter_num;

        // scatter to sink
        not_self_to_self = (d_scatter != d_end);
        raycast2<<<32,128>>>(d_scatter,d_end,&d_ray[rays_filled],scatter_num,end_num,not_self_to_self,d_tri_vertex,d_binned_triangles,d_tri_num_per_bin,end_num*scatter_num,&d_ray_index[rays_filled],source_num+end_num,source_num,num_bins,x_top_bottom,y_range,z_range);
        cudaDeviceSynchronize(); // Wait for the kernel to finish

        gpuErrchk( cudaGetLastError() );
        rays_filled += scatter_num * end_num;

        // scatter to scatter
            not_self_to_self = (d_scatter != d_scatter);
            raycast2<<<32,128>>>(d_scatter,d_scatter,&d_ray[rays_filled],scatter_num,scatter_num,not_self_to_self,d_tri_vertex,d_binned_triangles,d_tri_num_per_bin,scatter_num*scatter_num,&d_ray_index[rays_filled],source_num+end_num,source_num+end_num,num_bins,x_top_bottom,y_range,z_range);
            cudaDeviceSynchronize(); // Wait for the kernel to finish

            gpuErrchk( cudaGetLastError() );
        
    */
    // free unneeded arrays
    cudaMemcpy(h_scattering_network,d_scattering_network, scattering_network_size, cudaMemcpyDeviceToHost);
    gpuErrchk( cudaGetLastError() );

    cudaFree(d_source);
    cudaFree(d_end);
    cudaFree(d_points);
    cudaFree(d_scattering_network);
    cudaFree(d_ray);
    cudaFree(d_ray_index);

    
    gpuErrchk( cudaGetLastError() );
    


}

