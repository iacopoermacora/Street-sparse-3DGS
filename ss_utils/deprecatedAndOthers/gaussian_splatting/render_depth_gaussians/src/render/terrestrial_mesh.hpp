/*
 * terrestrial_mesh.hpp
 *
 *  Created on: Jan 19, 2015
 *      Author: jbr
 */

#ifndef TERRESTRIAL_MESH_
#define TERRESTRIAL_MESH_

#include <vector>

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <opencv2/core/core.hpp>

namespace rdc
{

struct TerrestrialMesh
{
    TerrestrialMesh
    (
        RTCDevice device,
        cv::Point3d offset,
        const std::vector<cv::Point3f>& vertices,
        const std::vector<int>& indices
    );

    TerrestrialMesh() = delete;
    TerrestrialMesh(const TerrestrialMesh&) = delete;
    TerrestrialMesh& operator=(const TerrestrialMesh&) = delete;

    ~TerrestrialMesh();

public:
    cv::Point3d mesh_offset;
    RTCScene scene;
    unsigned int geometry_id;
    std::vector<bool> valid_primitive;
};

} // namespace rdc

#endif /* TERRESTRIAL_MESH_ */
