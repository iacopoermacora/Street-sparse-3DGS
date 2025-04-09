project(json_rds_to_stations)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CMAKE_MODULE_PATH}")

include(select_toolchain)
include(target_functions)

# List all required dependencies
# Based on your includes, you need cf_io_recording and cf_date_time
# The third parameter is for "searchable packages" which might include boost
CfAddExecutableAndUnitTests(json_rds_to_stations "" "cf")