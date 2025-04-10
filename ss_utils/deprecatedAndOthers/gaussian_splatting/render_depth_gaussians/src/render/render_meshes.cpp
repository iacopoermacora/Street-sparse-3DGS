/*
 * render_mesh.cpp
 *
 *  Created on: Jan 5, 2015
 *      Author: jbr
 */

#include "render_meshes.hpp"

#include <cf/io/recording/helper_camera.hpp>
#include <cf/io/recording/recording.hpp>
#include <glog/logging.h>
#include <tbb/tbb.h>


const int kFaceSizei = 512 * 3;


namespace rdc
{

TerrestrialMeshesRenderer::TerrestrialMeshesRenderer(RTCDevice device) : device_(device)
{

}

TerrestrialMeshesRenderer::~TerrestrialMeshesRenderer()
{

}

// Creates a direction matrix for a direction.
// note: point_dir should be normalized
cv::Matx33f DirectionMatrix(const cv::Vec3f &point_dir)
{
    // make point_dir the new y axis

    const cv::Vec3f up(0.0f, 0.0f, 1.0f);

    cv::Vec3f new_y = point_dir;

    // new x axis:
    cv::Vec3f new_x;

    if (fabs(fabs(up.dot(point_dir)) - 1.0f) < 0.00001f) // if the direction is straight up or down...
    {
        new_x = cv::Vec3f(1.0f, 0.0f, 0.0f); // just take the actual x axis
    }
    else
    {
        new_x = cv::normalize(new_y.cross(up));
    }

    // new z axis:
    cv::Vec3f new_z = cv::normalize(new_x.cross(new_y));

    cv::Matx33f r = cv::Matx33f::zeros();

    for (int j = 0; j < 3; j++)
    {
        r(j, 0) = new_x(j);
        r(j, 1) = new_y(j);
        r(j, 2) = new_z(j);
    }

    return r;
}

void RenderScene(RTCDevice device_, RTCScene scene, cv::Matx33f p, const std::vector<std::vector<bool>*>& valid_triangle_index, cv::Mat* dist_face)
{
    VLOG(3) << "Render scene face";

    struct RTCORE_ALIGN(32) ValidMask8
    {
        unsigned int32_t[8] = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
    };

    ValidMask8 mask;

    int w = dist_face->cols;
    int h = dist_face->rows;

    float f = static_cast<float>(w) * 0.5f;

    tbb::parallel_for(0, h, [&](int iy)
    {
        float fy = f - 0.5f - static_cast<float>(iy);
        float* row_dist = dist_face->ptr<float>(iy);

        float fx = -f + 0.5f;
        for (int ix = 0; ix < w; ix += 8)
        {
            RTCRay8 ray;

            for (int r = 0; r < 8; ++r)
            {
                cv::Vec3f rdir(fx + static_cast<float>(r), f, fy);
                rdir = p * rdir;
                rdir /= cv::norm(rdir);

                ray.orgx[r] = 0.0f;
                ray.orgy[r] = 0.0f;
                ray.orgz[r] = 0.0f;

                ray.dirx[r] = rdir[0];
                ray.diry[r] = rdir[1];
                ray.dirz[r] = rdir[2];

                ray.tnear[r] = 0.0f;
                ray.tfar[r] = 1000.0f;
                ray.geomID[r] = RTC_INVALID_GEOMETRY_ID;
                ray.primID[r] = RTC_INVALID_GEOMETRY_ID;
                ray.mask[r] = -1;
                ray.time[r] = 0.0f;
            }

            rtcIntersect8(&mask, scene, ray);

            for (int r = 0; r < 8; ++r)
            {
                bool valid = ray.geomID[r] != RTC_INVALID_GEOMETRY_ID &&
                        (*valid_triangle_index[ray.instID[r]])[ray.primID[r]]; // property lookup

                row_dist[ix + r] = valid ? ray.tfar[r] : 0.0f;
            }
            fx += 8.0f;
        }
    });

    VLOG(3) << "Render scene face done";
}


template<typename T>
cv::Matx<T, 4, 4> ToMatx44(const cv::Matx<T, 3, 3>& in)
{
    return cv::Matx<T, 4, 4>
    (
        in(0,0), in(0,1), in(0,2), 0,
        in(1,0), in(1,1), in(1,2), 0,
        in(2,0), in(2,1), in(2,2), 0,
              0,       0,       0, 1

    );
}

template<typename T>
cv::Matx<T, 4, 4> MakeTranslationMatx(const cv::Point3_<T>& v)
{
    return cv::Matx<T, 4, 4>
    (
        1, 0, 0, v.x,
        0, 1, 0, v.y,
        0, 0, 1, v.z,
        0, 0, 0, 1
    );
}

cv::Matx44f MakeViewMatx44f(const cv::Matx33f& camera_rotation, const cv::Point3f& relative_camera_pos)
{
    const cv::Matx33f R_t = camera_rotation.t();
    return ToMatx44(R_t) * MakeTranslationMatx(-relative_camera_pos);
}

cv::Vec3f combineMixedAngles(float angleInDegrees, float angleInRadians) {
    // Convert degrees to radians
    float angleInRadians1 = angleInDegrees * M_PI / 180.0f;
    
    // Now both angles are in radians, so we can combine them
    float resultAngle = angleInRadians1 + angleInRadians;
    
    // Convert to a unit vector
    float x = std::sin(resultAngle);
    float y = std::cos(resultAngle);

    // Keep just the last 6 decimal places and do not allow the value -0
    x = std::round(x * 1000000) / 1000000;
    y = std::round(y * 1000000) / 1000000;
    if (x == -0) x = 0;
    if (y == -0) y = 0;
    
    // Return as Vec3f (z = 0 for 2D direction)
    return cv::Vec3f(x, y, 0);
}

void TerrestrialMeshesRenderer::Render
(
    const cf::io::RecordingExtrinsics& tr,
    const std::vector<std::shared_ptr<TerrestrialMesh> >& meshes,
    cv::Mat cube_distance[],
    float driving_direction,
    const std::string& directions_config = "3"  // Default to the 10-face configuration
)
{
    int width = kFaceSizei;
    int height = kFaceSizei;

    cv::Matx33f cam_rotation;
    cf::io::RecordingSourceTraits<cf::io::RecordingSource::TERRESTRIAL>::OrientationConvention::GetCameraRotation(tr, &cam_rotation);

    std::vector<cv::Matx44f> vs;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
        cv::Point3f t
        (
            static_cast<float>(tr.position.x - meshes[i]->mesh_offset.x),
            static_cast<float>(tr.position.y - meshes[i]->mesh_offset.y),
            static_cast<float>(tr.position.z - meshes[i]->mesh_offset.z)
        );

        vs.push_back(MakeViewMatx44f(cam_rotation, t));
    }

    // build a scene for rendering
    RTCScene scene = rtcDeviceNewScene(device_, RTC_SCENE_STATIC | RTC_SCENE_COHERENT | RTC_SCENE_COMPACT, RTC_INTERSECT8);

    std::vector<std::vector<bool>*> valid_triangle_index;
    // Build the scene
    for (size_t mesh_idx = 0; mesh_idx < meshes.size(); ++mesh_idx)
    {
        cv::Matx44f mv = vs[mesh_idx];
        unsigned int instance = rtcNewInstance(scene, meshes[mesh_idx]->scene);
        valid_triangle_index.push_back(&(meshes[mesh_idx]->valid_primitive));
        rtcSetTransform(scene, instance, RTC_MATRIX_ROW_MAJOR, (float*)&mv.val[0]);
    }

    rtcCommit(scene);

    // Define the number of faces and their directions based on the configuration
    int num_faces = 10;  // Default
    std::vector<float> directions;
    
    if (directions_config == "1") {  // F1R1B1L1 (4 faces)
        num_faces = 4;
        directions = {0.0f, 90.0f, 180.0f, 270.0f};  // Front, Right, Back, Left
    } 
    else if (directions_config == "2") {  // F1F2R1R2B1B2L1L2 (8 faces)
        num_faces = 8;
        directions = {0.0f, 45.0f, 90.0f, 135.0f, 180.0f, 225.0f, 270.0f, 315.0f};  // Two each of Front, Right, Back, Left
    }
    else if (directions_config == "3") {  // F1F2R1R2B1B2L1L2U1U2 (10 faces)
        num_faces = 10;
        directions = {0.0f, 45.0f, 90.0f, 135.0f, 180.0f, 225.0f, 270.0f, 315.0f, 45.0f, 225.0f};  // Original configuration
    }
    else if (directions_config == "4") {  // F1R1B1L1U1U2 (6 faces)
        num_faces = 6;
        directions = {0.0f, 90.0f, 180.0f, 270.0f, 45.0f, 225.0f};  // Front, Right, Back, Left, Up1, Up2
    }

    driving_direction = driving_direction - tr.orientation.z;
    // Create combined directions vector for the faces
    std::vector<cv::Vec3f> combined_directions(num_faces);
    for (int i = 0; i < num_faces; ++i) {
        combined_directions[i] = combineMixedAngles(directions[i], driving_direction);
    }

    for (int face_idx = 0; face_idx < num_faces; ++face_idx)
    {
        cv::Matx33f dir = DirectionMatrix(combined_directions[face_idx]);

        cube_distance[face_idx].create(cv::Size(width, height), CV_32FC1);

        RenderScene(device_, scene, dir, valid_triangle_index, &cube_distance[face_idx]);
    }
}

} // namespace rdc
