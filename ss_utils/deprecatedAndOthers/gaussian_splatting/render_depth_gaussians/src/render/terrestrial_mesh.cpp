/*
 * terrestrial_mesh.cpp
 *
 *  Created on: Jan 19, 2015
 *      Author: jbr
 */

#include "terrestrial_mesh.hpp"

#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

DEFINE_double
(
    max_valid_triangle_area,
    0.5,
    "The maximum area (in m2) of valid triangles. Larger triangles lead to unknown depth values."
);

DEFINE_double
(
    max_valid_triangle_side_length,
    0.60,
    "The maximum length of any side of valid triangles. Larger triangles lead to unknown depth values."
);

namespace rdc
{


TerrestrialMesh::TerrestrialMesh
(
    RTCDevice device,
    cv::Point3d offset,
    const std::vector<cv::Point3f>& vertices,
    const std::vector<int>& indices
) : mesh_offset(offset),
    valid_primitive(indices.size()/3)
{
    // create a static scene to contain the geometry
    scene = rtcDeviceNewScene(device, RTC_SCENE_STATIC | RTC_SCENE_COHERENT | RTC_SCENE_COMPACT, RTC_INTERSECT8);
    geometry_id = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, indices.size() / 3, vertices.size(), 1);

    struct RVertex   { float x, y, z, a; };
    struct RTriangle { int v0, v1, v2; };

    RVertex* rvertices = (RVertex*) rtcMapBuffer(scene, geometry_id, RTC_VERTEX_BUFFER);

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        rvertices[i].x = vertices[i].x;
        rvertices[i].y = vertices[i].y;
        rvertices[i].z = vertices[i].z;
        rvertices[i].a = 1.0f;
    }

    rtcUnmapBuffer(scene, geometry_id, RTC_VERTEX_BUFFER);

    RTriangle* rtriangles = (RTriangle*) rtcMapBuffer(scene, geometry_id, RTC_INDEX_BUFFER);

    const double square_max_length = FLAGS_max_valid_triangle_side_length * FLAGS_max_valid_triangle_side_length;

    for (size_t i = 0; i < indices.size(); i += 3)
    {
        size_t tri_idx = i / 3;

        rtriangles[tri_idx].v0 = static_cast<int>(indices[i]);
        rtriangles[tri_idx].v1 = static_cast<int>(indices[i + 1]);
        rtriangles[tri_idx].v2 = static_cast<int>(indices[i + 2]);

        const cv::Point3f& a = vertices[indices[i]];
        const cv::Point3f& b = vertices[indices[i + 1]];
        const cv::Point3f& c = vertices[indices[i + 2]];

        // compute the area of the triangle and see if it is too big
        cv::Point3f ab = b - a;
        cv::Point3f ac = c - a;
        cv::Point3f bc = c - b;

        float area = 0.5f * cv::norm(ab.cross(ac));

        bool valid = area <= FLAGS_max_valid_triangle_area &&
                     ab.ddot(ab) <= square_max_length &&
                     ac.ddot(ac) <= square_max_length &&
                     bc.ddot(bc) <= square_max_length;

        valid_primitive[tri_idx] = valid;
    }

    rtcUnmapBuffer(scene, geometry_id, RTC_INDEX_BUFFER);

    VLOG(3) << "Start building mesh.";
    rtcCommit(scene);
    VLOG(3) << "Done building mesh.";
}

TerrestrialMesh::~TerrestrialMesh()
{
    rtcDeleteGeometry(scene, geometry_id);
    // delete the scene
    rtcDeleteScene(scene);
}

} // namespace
