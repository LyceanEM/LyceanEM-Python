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
# include <complex>
#include "em.cuh"
#include "raycasting.cuh"
#include "gpu_error.cuh"
#include "raycasting_accelerated.cuh"
#include "terrain_acceleration_build.cuh"
#include <utility>
#include <tuple>
#include <chrono>   
namespace py = pybind11;
int needed_bytes_raycaster(int scatter_depth, int end_num, int source_num, int scatter_num, int tri_num){
    int source_size = source_num * sizeof(float3);
    int end_size = end_num * sizeof(float3);
    int scatter_size = scatter_num * sizeof(float3);
    int ray_size = (source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num) * sizeof(float4);
    int ray_index_size = (source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num) * sizeof(int2);
    int tri_vertex_size = tri_num * sizeof(float3);
    int triangle_index_size = tri_num * sizeof(int3);
    int raycaster_size = source_size + end_size + scatter_size + ray_size + ray_index_size + tri_vertex_size + triangle_index_size;
    return raycaster_size;
}
long needed_enchanced_raycasting(int scatter_depth, int end_num, int source_num, int scatter_num, int tri_num,int2 tile_num){
    long source_size = source_num * sizeof(float3);
    long bin_count_size = tile_num.x * tile_num.y * sizeof(int);
    long end_size = end_num * sizeof(float3);
    long scatter_size = scatter_num * sizeof(float3);
    long ray_size = (source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num) * sizeof(float4);
    long ray_index_size = (source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num) * sizeof(int2);
    long tri_vertex_size = tri_num * sizeof(float3);
    long triangle_index_size = tri_num * 4 * sizeof(int3);
    long raycaster_size = source_size + end_size + scatter_size + ray_size + ray_index_size + tri_vertex_size + triangle_index_size + bin_count_size;
    long scatter_network_size = (end_num*source_num) * sizeof(complex_float3);

    return raycaster_size;
}
int needed_bytes_em(int scatter_depth, int end_num, int source_num, int scatter_num){
    int ray_num = source_num * end_num+ source_num * scatter_num + scatter_num * end_num + scatter_num * scatter_num;
    int ray_size = ray_num * sizeof(float4);
    int ray_index_size = ray_num * sizeof(int2);
    int points_size = (source_num+scatter_num+end_num) * sizeof(PointData);
    int scatter_network_size = (end_num*source_num+scatter_depth) * sizeof(complex_float3);
    int em_size = ray_size + ray_index_size + points_size + scatter_network_size;
    return em_size;


}

long input_bytes(int source_num, int end_num,int scatter_num, int tri_vertex_num, int binned_triangles_num, int bin_count_num){
    long source_size = source_num * sizeof(float3);
    //std::cout<<"source size  "<<source_size<<std::en dl;
    long end_size = end_num * sizeof(float3);
    //std::cout<<"end size  "<<end_size<<std::endl;
    long scatter_size = scatter_num * sizeof(float3);
    //std::cout<<"scatter size  "<<scatter_size<<std::endl;
    long tri_vertex_size = tri_vertex_num * sizeof(float3);
    //std::cout<<"tri vertex size  "<<tri_vertex_size<<std::endl;
    long triangle_index_size = binned_triangles_num * sizeof(long3);
    //std::cout<<"triangle index size  "<<triangle_index_size<<std::endl;
    long bin_count_size = bin_count_num * sizeof(int);
    //std::cout<<"bin count size  "<<bin_count_size<<std::endl;
    long points_size = (source_num+scatter_num+end_num) * sizeof(PointData);
    //std::cout<<"points size  "<<points_size<<std::endl;
    long input_size = source_size + end_size + scatter_size + tri_vertex_size + triangle_index_size + bin_count_size + points_size;
    return input_size;
}



py::array_t<std::complex<float>> calculate_scattering(py::array_t<float> source, py::array_t<float> end, 
                                                        py::array_t<float> triangle_vertex, 
                                                        float wave_length ,
                                                        py::array_t<float> ex_real, py::array_t<float> ex_imag, py::array_t<float> ey_real, py::array_t<float> ey_imag, py::array_t<float> ez_real, py::array_t<float> ez_imag, 
                                                        py::array_t<bool> is_electric, py::array_t<float> permittivity_real, py::array_t<float> permittivity_imag, py::array_t<float> permeability_real, py::array_t<float> permeability_imag,
                                                        py::array_t<float> normal, float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, float tile_size, py::array_t<int> bin_count, py::array_t<int> bin_triangles, int n_cellsx, int n_cellsy, int binned_tri_num){

    // validate data about input arrays
    if (source.ndim() != 2 || source.shape(1) != 3){throw std::runtime_error("source must be a 2D array with 3 columns");}
    if (end.ndim() != 2 || end.shape(1) != 3){throw std::runtime_error("end must be a 2D array with 3 columns");}
    if (triangle_vertex.ndim() != 2 || triangle_vertex.shape(1) != 3){throw std::runtime_error("triangle_vertex must be a 2D array with 3 columns");}
    if (ex_real.ndim() != 1){throw std::runtime_error("ex_real must be a 1D array");}
    if (ex_imag.ndim() != 1){throw std::runtime_error("ex_imag must be a 1D array");}
    if (ey_real.ndim() != 1){throw std::runtime_error("ey_real must be a 1D array");}
    if (ey_imag.ndim() != 1){throw std::runtime_error("ey_imag must be a 1D array");}
    if (ez_real.ndim() != 1){throw std::runtime_error("ez_real must be a 1D array");}
    if (ez_imag.ndim() != 1){throw std::runtime_error("ez_imag must be a 1D array");}
    if (is_electric.ndim() != 1){throw std::runtime_error("is_electric must be a 1D array");}
    if (permittivity_real.ndim() != 1){throw std::runtime_error("permittivity_real must be a 1D array");}
    if (permittivity_imag.ndim() != 1){throw std::runtime_error("permittivity_imag must be a 1D array");}
    if (permeability_real.ndim() != 1){throw std::runtime_error("permeability_real must be a 1D array");}
    if (permeability_imag.ndim() != 1){throw std::runtime_error("permeability_imag must be a 1D array");}
    if (normal.ndim() != 2 || normal.shape(1) != 3){throw std::runtime_error("normal must be a 2D array with 3 columns");}
    if (bin_count.ndim() != 1){throw std::runtime_error("bin_count must be a 1D array");}
    if (bin_triangles.ndim() != 1){throw std::runtime_error("bin_triangles must be a 1D array");}
    if (bin_count.shape(0) != n_cellsx * n_cellsy){throw std::runtime_error("bin_count must have n_cellsx * n_cellsy elements");}
    if (bin_triangles.shape(0) != binned_tri_num*3 ){throw std::runtime_error("bin_triangles 3* bintrinum elements");}

                                                    auto fisttttt = std::chrono::high_resolution_clock::now();
    //get the size of the input arrays
    int source_size = source.shape(0);
    int end_size = end.shape(0);
    int tri_vertex_size = triangle_vertex.shape(0);



    if (ex_real.shape(0) != ex_imag.shape(0) ||
        ex_real.shape(0) != ey_real.shape(0) ||
        ex_real.shape(0) != ey_imag.shape(0) ||
        ex_real.shape(0) != ez_real.shape(0) ||
        ex_real.shape(0) != ez_imag.shape(0) ||
        ex_real.shape(0) != is_electric.shape(0) ||
        ex_real.shape(0) != permittivity_real.shape(0) ||
        ex_real.shape(0) != permittivity_imag.shape(0) ||
        ex_real.shape(0) != permeability_real.shape(0) ||
        ex_real.shape(0) != permeability_imag.shape(0) ||
        ex_real.shape(0) != normal.shape(0)||
        ex_real.shape(0) != source_size)
    {
        throw std::runtime_error("All point data arrays must have the same size = source_size + end_size + scatter_size");
    }

    //get the pointers to the data
    float* source_ptr = (float*) source.request().ptr;
    float* end_ptr = (float*) end.request().ptr;
    float* triangle_vertex_ptr = (float*) triangle_vertex.request().ptr;
    float* ex_real_ptr = (float*) ex_real.request().ptr;
    float* ex_imag_ptr = (float*) ex_imag.request().ptr;
    float* ey_real_ptr = (float*) ey_real.request().ptr;
    float* ey_imag_ptr = (float*) ey_imag.request().ptr;
    float* ez_real_ptr = (float*) ez_real.request().ptr;
    float* ez_imag_ptr = (float*) ez_imag.request().ptr;
    bool* is_electric_ptr = (bool*) is_electric.request().ptr;
    float* permittivity_real_ptr = (float*) permittivity_real.request().ptr;
    float* permittivity_imag_ptr = (float*) permittivity_imag.request().ptr;
    float* permeability_real_ptr = (float*) permeability_real.request().ptr;
    float* permeability_imag_ptr = (float*) permeability_imag.request().ptr;
    float* normal_ptr = (float*) normal.request().ptr;
    int* bin_count_ptr = (int*) bin_count.request().ptr;
    int* bin_triangles_ptr = (int*) bin_triangles.request().ptr;
    //printf("source_size %d\n",source_size);

    std::cout<< "source size  "<<source_size<<std::endl;
    int3* d_binned_triangles;
    int2* d_bin_count;
    float3* d_tri_vertex;
    std::vector<int2> binned_num(n_cellsx * n_cellsy);
    std::cout<< "source size  "<<source_size<<std::endl;
    int sum = 0;
    for (int i = 0; i < n_cellsx * n_cellsy; i++){
        binned_num[i] = make_int2( bin_count_ptr[i],sum);
        sum += bin_count_ptr[i];    
    }

    int2* bin_count_ptr2 = binned_num.data();
    //print free toral memory
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<< "free  "<<free<<std::endl;
    std::cout<< "total  "<<total<<std::endl;


    cudaMalloc(&d_binned_triangles, binned_tri_num * sizeof(int3));
    cudaMalloc(&d_bin_count, n_cellsx * n_cellsy * sizeof(int2));
    cudaMalloc(&d_tri_vertex, tri_vertex_size * sizeof(float3));
    //gpuerror check
    cudaMemGetInfo(&free, &total);
    std::cout<< "free  "<<free<<std::endl;
    std::cout<< "total  "<<total<<std::endl;
    gpuErrchk( cudaGetLastError() );


    cudaMemcpy(d_binned_triangles, bin_triangles_ptr, binned_tri_num * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bin_count, bin_count_ptr2, n_cellsx * n_cellsy * sizeof(int2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tri_vertex, triangle_vertex_ptr, tri_vertex_size * sizeof(float3), cudaMemcpyHostToDevice);
    //gpuerror check
    gpuErrchk( cudaGetLastError() );
    printf("hi from post %d\n",source_size);

    std::vector<PointData> points_vec(source_size);
    PointData* points = points_vec.data();


    for (int i = 0; i < source_size; i++){
        points_vec[i] = create_point_data(ex_real_ptr[i], ex_imag_ptr[i], ey_real_ptr[i], ey_imag_ptr[i], ez_real_ptr[i], ez_imag_ptr[i], is_electric_ptr[i], permittivity_real_ptr[i], permittivity_imag_ptr[i], permeability_real_ptr[i], permeability_imag_ptr[i], normal_ptr[i*3], normal_ptr[i*3+1], normal_ptr[i*3+2]);
    }
    //pointer to the points
  




    int num_gpus;
    

    cudaGetDeviceCount( &num_gpus );
    int total_free = 0;
    std::vector<int> gpu_share (num_gpus);
    
    for (int i = 0; i < 1; i++){
        cudaSetDevice(i);
        cudaMemGetInfo(&free, &total);
        free *= 0.95;
        total_free += free;
        //std::cout<< "free  "<<free<<std::endl;
        gpu_share[i] = free;
    }
    //std::cout<< "total free  "<<total_free<<std::endl;
    //std::cout<< "source size  "<<total<<std::endl;
    long uuu = needed_enchanced_raycasting(0,1E6,2.5E5,0,1E6,make_int2(100,100));
    //std::cout<< "source size  "<<uuu<<std::endl;
    //std::cout<< "source size  "<<uuu/total_free<<std::endl;
    std::cout<<"Hi from pre bduild"<<std::endl;

    


    // declare numpy complex array
    py::array_t<std::complex<float>> scattering_network_py = py::array_t<std::complex<float>>(source_size * end_size *3);
    std::cout<<"Hi from post declare numpy complex array"<<std::endl;
    std::complex<float>* scattering_network_py_ptr = (std::complex<float>*) scattering_network_py.request().ptr;

    std::cout<<"Hi from post getting pointer to numpy complex array"<<std::endl;
    int source_index = 0;
    //std::cout<< "chunk number  "<<chunk_number<<std::endl;
   // std::cout<< "g  "<<num_gpus<<std::endl;
    /*
    for (int j = 0; j < chunk_number; j++){
        for (int i = 0; i < num_gpus; i++){
        
        
            cudaSetDevice(i);
            
            int source_size_gpu = (i == 0) ? ((gpu_share[i] / total_free) * source_size_temp + 1) : ((gpu_share[i] / total_free) * source_size_temp);
            std::vector<complex_float3> scattering_network(source_size_gpu * end_size);
            //pointer to the scattering network
            complex_float3* scattering_network_ptr = scattering_network.data();



            std::pair<float4*,int2*> rays = raycast_wrapper(&source_ptr[source_index], end_ptr, scatter_ptr, source_size_gpu, end_size, scatter_size, triangle_vertex_ptr, triangles_ptr, triangle_size);
            frequency_wrapper( scatter_depth, rays.first, points, rays.second, wave_length, source_size_gpu, scatter_size, end_size,scattering_network_ptr);
            
            for (int i = 0; i < scattering_network.size() ; i++){
                ////std::cout<< "indexes  "<<source_index+i*3<<"  "<<source_size * end_size *3<<std::endl;


            }
            source_index += source_size_gpu;
        }








    }*/
    //std::cout<<"hi"<<std::endl;
    //std::cout<<"source size  "<<source_size<<"end size  "<<end_size<<std::endl;
    std::vector<complex_float3> scattering_network(source_size* end_size);
    std::cout<<"Hi from post declare scattering network"<<std::endl;
//std::cout<<"triangle size  "<<triangle_size<<std::endl;
//std::cout<<"triangle vertex size  "<<tri_vertex_size<<std::endl;
            //pointer to the scattering network
            complex_float3* scattering_network_ptr = scattering_network.data();


            //std::cout<<"hi1"<<numbuytes<<std::endl;


            //std::tuple<int3*,int*,float3*> grid_structure = grid_structure_builder_wrapper(triangle_vertex_ptr, triangles_ptr, triangle_size, make_int2(tile_numy,tile_numz), make_float2(xmin,xmax), make_float2(ymin,tile_size), make_float2(zmin,tile_size), tri_vertex_size);
            //std::cout<<"hi2"<<std::endl;

            //int3* d_binned_triangles = std::get<0>(grid_structure);
            

            //std::cout<<"hi3"<<std::endl;
            //std::cout<<"time taken for raycast_wrapper2  "<<std::chrono::duration_cast<std::chrono::milliseconds>(begin_time2 - end_time).count() << "ms"<<std::endl;


           // std::pair<float4*,int2*> rays = raycast_wrapper(source_ptr, end_ptr, scatter_ptr, source_size, end_size, scatter_size, triangle_vertex_ptr, triangles_ptr, triangle_size);

            //frequency_wrapper( 0, rays.first, points, rays.second, wave_length, source_size, scatter_size, end_size,scattering_network_ptr);
            //std::cout << "Time taken for frequency_wrapper: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time2 - begin_time2).count() << "ms" << std::endl;
            //std::cout<<"hi4"<<std::endl;
            auto now = std::chrono::high_resolution_clock::now();
             raycast_wrapper2(&source_ptr[source_index], end_ptr, source_size, end_size,d_tri_vertex, d_binned_triangles, d_bin_count
            ,make_int2(n_cellsx,n_cellsy) , make_float2(xmin,xmax), make_float2(ymin,tile_size), make_float2(zmin,tile_size),points, wave_length,scattering_network_ptr);

            auto end_time = std::chrono::high_resolution_clock::now();
            std::cout<<"time taken for raycast_wrapper2  "<<std::chrono::duration_cast<std::chrono::milliseconds>(end_time - now).count() << "ms"<<std::endl;



            for (int i = 0; i < scattering_network.size() ; i++){
             scattering_network_py_ptr[i*3+0] = std::complex<float>(scattering_network_ptr[i].x.x, scattering_network_ptr[i].x.y);
                scattering_network_py_ptr[i*3+1] = std::complex<float>(scattering_network_ptr[i].y.x, scattering_network_ptr[i].y.y);
                scattering_network_py_ptr[i*3+2] = std::complex<float>(scattering_network_ptr[i].z.x, scattering_network_ptr[i].z.y);


                
            }

    //std::cout<< "source size  "<<source_size<<std::endl;
    //std::cout<< "end size  "<<end_size<<std::endl;
    //std::cout<< "scatter size  "<<scatter_size<<std::endl;
    //std::cout<< "triangle size  "<<triangle_size<<std::endl;
    //std::cout << "Time taken for the whole function: " << std::chrono::duration_cast<std::chrono::milliseconds>(lasttttt - fisttttt).count() << "ms" << std::endl;
    //std::cout<< "time taken post build till coppierd  "<<std::chrono::duration_cast<std::chrono::milliseconds>(lasttttt - end_time).count() << "ms"<<std::endl;
    cudaFree(d_binned_triangles);
    cudaFree(d_bin_count);
    cudaFree(d_tri_vertex);


    return scattering_network_py;
}


PYBIND11_MODULE(em, m) {
    m.def("calculate_scattering", &calculate_scattering, "Calculate scattering");
    m.def("bin_counts_to_numpy", &bin_counts_to_numpy);
    m.def("bin_triangles_to_numpy", &bin_triangles_to_numpy);

}


