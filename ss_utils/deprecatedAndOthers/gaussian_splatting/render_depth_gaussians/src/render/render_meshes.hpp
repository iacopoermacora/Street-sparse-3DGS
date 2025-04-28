/*
 * render_mesh.hpp
 *
 *  Created on: Jan 5, 2015
 *      Author: jbr
 */

#ifndef RENDER_MESHES_HPP_
#define RENDER_MESHES_HPP_


#include <boost/noncopyable.hpp>
#include <boost/shared_ptr.hpp>
#include <cf/io/recording/recording.hpp>
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <opencv2/core/core.hpp>

#include "terrestrial_mesh.hpp"

namespace rdc
{

class TerrestrialMeshesRenderer
{
public:
    TerrestrialMeshesRenderer(RTCDevice device);
    ~TerrestrialMeshesRenderer();

    void Render
    (
        const cf::io::RecordingExtrinsics& tr,
        const std::vector<std::shared_ptr<TerrestrialMesh> >& meshes,
        cv::Mat cube_distance[],
        float driving_direction,
        const std::string& directions_config = "3"
    );

private:

    RTCDevice device_;
};

} // namespace

#endif /* RENDER_MESHES_HPP_ */