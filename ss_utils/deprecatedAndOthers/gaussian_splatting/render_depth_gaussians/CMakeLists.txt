project(render_depth_gaussians)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_MODULE_PATH}")

include(select_toolchain)
include(target_functions)

CfAddExecutable(render_depth_gaussians "las;tbb;embree" "cf;cf_lidar;cf_3d")


