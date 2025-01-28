#pragma once

#include "vector_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "math_utility.cuh"
#include "gpu_error.cuh"
#include "em.cuh"
#include "raycasting.cuh"


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
    return point;

}
__device__ __inline__ int2 bin_point_2d( float2 y_range, float2 z_range, float3 point){

    int biny = floorf((point.y-y_range.x)/y_range.y);
    int binz = floorf((point.z-z_range.x)/z_range.y);
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







__device__ __inline__ bool intersection_tiles(int i, float4 *ray, float3 *tri_vertex,int3* triangle_index, int tri_num, float3 origin,int end_num,int flag)
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
     int2 num_bins, int end_num, int flag, float2 x_top_bottom, float2 y_range, float2 z_range)
{
    float4 ray_vector = rays[ray_index];
    float3 ray_direction = make_float3(ray_vector.x,ray_vector.y,ray_vector.z);
    int c = ray_index / end_num;
    int d = ray_index % end_num;

    float3 ray_origin = source[c];
    float3 endss = end[d];

    // are the ends of the ray above below or inside the structure
    bool origin_above_sink = (ray_origin.x>endss.x);

    bool origin_above_structure = (ray_origin.x>x_top_bottom.y);
    bool origin_below_structure = (ray_origin.x<x_top_bottom.x);
    bool origin_inside_structure = (!origin_above_structure && !origin_below_structure);


    bool sink_above_structure = (endss.x>x_top_bottom.y);
    bool sink_below_structure = (endss.x<x_top_bottom.x);
    bool sink_inside_structure = (!sink_above_structure && !sink_below_structure);

    int2 tile_origin;
    int2 tile_destination;
    float2 origin_in_fractional_bin_units;
    if(origin_above_sink){
        //FULL RAYCASTING
        if(origin_above_structure && sink_below_structure){
        float3 top_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.y,y_range.x,z_range.x));
        float3 bottom_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.x,y_range.x,z_range.x));
        origin_in_fractional_bin_units = make_float2((top_intersection.y-y_range.x)/y_range.y,(top_intersection.z-z_range.x)/z_range.y);
        tile_origin = bin_point_2d(y_range,z_range,top_intersection);
        tile_destination = bin_point_2d(y_range,z_range,bottom_intersection);

        }
        //TERMINATE EARLY inside
        else if(origin_above_structure && sink_inside_structure){
        float3 top_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.y,y_range.x,z_range.x));
        origin_in_fractional_bin_units = make_float2((top_intersection.y-y_range.x)/y_range.y,(top_intersection.z-z_range.x)/z_range.y);
        tile_origin = bin_point_2d(y_range,z_range,top_intersection);
        tile_destination = bin_point_2d(y_range,z_range,endss);

        }
        //start inside and exit
        else if(sink_below_structure && origin_inside_structure){
        float3 bottom_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.x,y_range.x,z_range.x));
        origin_in_fractional_bin_units = make_float2((ray_origin.y-y_range.x)/y_range.y,(ray_origin.z-z_range.x)/z_range.y);
        tile_origin = bin_point_2d(y_range,z_range,ray_origin);
        tile_destination = bin_point_2d(y_range,z_range,bottom_intersection);

        }
        // start and finish inside
        else if(sink_inside_structure && origin_inside_structure){
            origin_in_fractional_bin_units = make_float2((ray_origin.y-y_range.x)/y_range.y,(ray_origin.z-z_range.x)/z_range.y);
            tile_origin = bin_point_2d(y_range,z_range,ray_origin);
            tile_destination = bin_point_2d(y_range,z_range,endss);
        }

        else{
            ray_point_index[ray_index] = make_int2(c,d);
            return;
        }
      
        

    }
    else{
        //FULL RAYCASTING
        if(origin_below_structure && sink_above_structure){
        float3 top_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.y,y_range.x,z_range.x));
        float3 bottom_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.x,y_range.x,z_range.x));
        origin_in_fractional_bin_units = make_float2((bottom_intersection.y-y_range.x)/y_range.y,(bottom_intersection.z-z_range.x)/z_range.y);
        tile_origin = bin_point_2d(y_range,z_range,bottom_intersection);
        tile_destination = bin_point_2d(y_range,z_range,top_intersection);

    }
        //TERMINATE EARLY inside
        else if(origin_below_structure && sink_inside_structure){
        float3 bottom_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.x,y_range.x,z_range.x));
        tile_origin = bin_point_2d(y_range,z_range,bottom_intersection);
        origin_in_fractional_bin_units = make_float2((bottom_intersection.y-y_range.x)/y_range.y,(bottom_intersection.z-z_range.x)/z_range.y);
        tile_destination = bin_point_2d(y_range,z_range,endss);

        }
        //start inside
        else if(sink_above_structure && origin_inside_structure){
        float3 top_intersection = ray_plane_intersection(ray_direction, ray_origin, make_float3(1,0,0), make_float3(x_top_bottom.y,y_range.x,z_range.x));
        origin_in_fractional_bin_units = make_float2((ray_origin.y-y_range.x)/y_range.y,(ray_origin.z-z_range.x)/z_range.y);
        tile_origin = bin_point_2d(y_range,z_range,ray_origin);
        tile_destination = bin_point_2d(y_range,z_range,top_intersection);

        }
        // start and finish inside
        else if(sink_inside_structure && origin_inside_structure){
            origin_in_fractional_bin_units = make_float2((ray_origin.y-y_range.x)/y_range.y,(ray_origin.z-z_range.x)/z_range.y);
            tile_origin = bin_point_2d(y_range,z_range,ray_origin);
            tile_destination = bin_point_2d(y_range,z_range,endss);
        }
        else{
            ray_point_index[ray_index] = make_int2(c,d);
            return;
        }

    }






    

    // gradient of ray component wise
    float2 gradient = gradient_component_y_z_2d(ray_vector);

    ////printf("gradient %f %f %f %f \n",gradient.x,gradient.y,endss.y-ray_origin.y, endss.z-ray_origin.z);
    //what direction to step in x or y-direction (either +1 or -1)
    int2 step = step_direction(ray_vector);
    double2 distance_to_next_side;
    //check step lines ip with box order



   


    
    if(step.x == 1){
        distance_to_next_side.x = ((tile_origin.x + 1.0) - origin_in_fractional_bin_units.x) * gradient.x;
    }
    else{
        distance_to_next_side.x = (origin_in_fractional_bin_units.x - tile_origin.x) * gradient.x;
    }
    if(step.y == 1){
        distance_to_next_side.y = ((tile_origin.y + 1.0) - origin_in_fractional_bin_units.y) * gradient.y;
    }
    else{
        distance_to_next_side.y = (origin_in_fractional_bin_units.y - tile_origin.y) * gradient.y;
    }
    bool distance_to_next_side_is_positive = (distance_to_next_side.x >= 0 && distance_to_next_side.y >= 0);
    assert(distance_to_next_side_is_positive);

    bool arrived_at_destination = false;
    bool intersection_with_triangle = false;
    bool intersected = false;
    bool outside_map = (tile_origin.x < 0 || tile_origin.x >= num_bins.x || tile_origin.y < 0 || tile_origin.y >= num_bins.y);
    gradient.x/=2;
    gradient.y/=2;
    int x_ranfe = std::abs(tile_destination.x-tile_origin.x)+1;
    int y_ranfe = std::abs(tile_destination.y-tile_origin.y)+1;

    for( int i = 0; i < x_ranfe; i++){
            for( int j = 0; j < y_ranfe; j++){

                int cell_index = (i*step.x+tile_origin.x)* num_bins.y + j*step.y + tile_origin.y;
                outside_map = (i*step.x+tile_origin.x < 0 || i*step.x+tile_origin.x >= num_bins.x || j*step.y+tile_origin.y < 0 || j*step.y+tile_origin.y >= num_bins.y);
                if(!outside_map){

                    intersection_with_triangle = intersection_tiles(ray_index,
                                                            rays,
                                                            tri_vertex, 
                                                            (int3*)&(binned_triangles[tri_num_in_bin[cell_index].y]),
                                                            tri_num_in_bin[cell_index].x, 
                                                            ray_origin, 
                                                                end_num, flag);

                }
                arrived_at_destination = (i+tile_origin.x == tile_destination.x && j+tile_origin.y == tile_destination.y);
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
        ray_point_index[ray_index] = make_int2(c,d);
    }


    ////printf("cell_index %i\n",cell_index);
    

    


/*
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
            if(!arrived_at_destination && std::abs(difference_distance) < epsilon){
                
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
            if(!arrived_at_destination && std::abs(difference_distance) < epsilon){
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
            ray_point_index[ray_index] = make_int2(c,d);
        }
    
    }
    else
    {
        ray_point_index[ray_index] = make_int2(c,d);
    }
    */
    

}








__global__ void raycast_tiles(float3 *source, float3 *end, float4 *ray, int source_num, int end_num, int flag, float3 *tri_vertex,int3 *binned_triangles, int2* tri_num_in_bin, int ray_num,
     int2 *ray_index, int2 num_bins, float2 x_top_bottom, float2 y_range, float2 z_range,PointData* points, float wave_length, complex_float3* scattering_network, const float2 alpha_beta)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = thread; i < ray_num; i+=stride)
    {

        points_to_rays(i,source,end,ray,source_num,end_num,flag);

        DDA(source, end, ray,ray_index,i,binned_triangles,tri_vertex,tri_num_in_bin,num_bins,end_num,flag,x_top_bottom,y_range,z_range);
        if(ray_index[i].x != -1){
            complex_float3 ray_field = em_wave(alpha_beta, ray[i],points[ray_index[i].x],points[ray_index[i].y],wave_length);
            scattering_network[i] = ray_field;
        }
    }

}

__global__ void set_values(int2 *ray_index, int ray_num, int2 value,int end_num)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;  
    for(int i = thread; i < ray_num; i+=stride)
    {
        int c = i / end_num;
        int d = i % end_num;
        ray_index[i] = make_int2(c,d);

    }
}





void raycast_wrapper_tiles (float *source, float *end, int source_num, int end_num, float3 *d_tri_vertex,int3 *d_binned_triangles, int2* d_tri_num_per_bin, int2 num_bins, float2 x_top_bottom, float2 y_range, float2 z_range,
    PointData* points, float wave_length, complex_float3* h_scattering_network, float2 alpha_beta)
{
    // declare device memory
    float3 *d_source;
    float3 *d_end;

    float3 *d_scatter;
    float4 *d_ray;
    int2 *d_ray_index;



    // calculate size of arrays
    int source_size = source_num * sizeof(float3);
    int end_size = end_num * sizeof(float3);

    int ray_size = (source_num * end_num) * sizeof(float4);
    int ray_index_size = (source_num * end_num) * sizeof(int2);

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
    set_values<<<32,256>>>(d_ray_index, (source_num * end_num),make_int2(-1,-1),end_num);
    gpuErrchk( cudaGetLastError() );


 
    //launch kernel for each ray_wave

    // source to sink
    bool not_self_to_self = (d_source != d_end);
    PointData* d_points;
    complex_float3* d_scattering_network;

    // calculate size of arrays
    int points_size = (source_num+end_num) * sizeof(PointData);

    int scattering_network_size = (end_num*source_num) * sizeof(complex_float3);

    
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

    raycast_tiles<<<32,256>>>(d_source,d_end,d_ray,source_num,end_num,not_self_to_self,d_tri_vertex,d_binned_triangles,d_tri_num_per_bin,source_num*end_num, d_ray_index,num_bins,x_top_bottom,y_range,z_range,d_points,wave_length,d_scattering_network, alpha_beta);
    //get last error
   
    gpuErrchk( cudaGetLastError() );

    //printer<<<32,128>>>(d_ray_index,source_num*end_num);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

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

