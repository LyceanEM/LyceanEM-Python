import numpy as np
import cupy as cp
import optix as ox

from lyceanem.base import structures

class stuctures_optix(structures):
    """
    Alteration of the structures class to output the solid triangle mesh to an optix primative triangle structure
    """
    def create_basic_acceleration_structure(self,ctx):
        """
        create triangle acceleratoin structure on the specified Optix device context
        Parameters
        ----------
        ctx : Optix Device Context

        Returns
        -------
        acceleration_structure : optix acceleration structure

        """
        vertices=np.empty((0,3),dtype=np.float32)
        for item in self.solids:
            vertices = np.append(vertices, np.asarray(item.vertices))
        build_input=ox.BuildInputTriangleArray(vertices,flags=[ox.GeometryFlags.NONE])
        acceleration_structure=ox.AccelerationStructure(ctx,build_input,compact=True)
        return acceleration_structure


def create_module(ctx, pipeline_opts):
    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.FULL, opt_level=ox.CompileOptimizationLevel.LEVEL_0)
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module


def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                              entry_function_CH="__closesthit__ch")

    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx, program_grps, pipeline_options):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1,
                                       debug_level=ox.CompileDebugLevel.FULL)

    pipeline = ox.Pipeline(ctx,
                           compile_options=pipeline_options,
                           link_options=link_opts,
                           program_groups=program_grps)

    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 1)  # max_dc_depth

    return pipeline


def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp, names=('rgb',), formats=('3f4',))
    miss_sbt['rgb'] = [0.3, 0.1, 0.2]

    hit_sbt = ox.SbtRecord(hit_grp)
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)

    return sbt

def log_callback(level, tag, msg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, msg))
    pass


#use test structure
environment=structures_optix()
#first step creates the cuda context
ctx = ox.DeviceContext(validation_mode=True, log_callback_function=log_callback, log_callback_level=3)
#then create acceleration structure
acceleration_structure= environment.create_basic_acceleration_structure(ctx)
#setup pipline option, implying a single acceleration structure
pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                 num_payload_values=3,
                                                 num_attribute_values=3,
                                                 exception_flags=ox.ExceptionFlags.NONE,
                                                 pipeline_launch_params_variable_name="params")
#compile module
module = create_module(ctx, pipeline_options)
program_grps = create_program_groups(ctx, module)
pipeline = create_pipeline(ctx, program_grps, pipeline_options)
sbt = create_sbt(program_grps)
img = launch_pipeline(pipeline, sbt, gas)