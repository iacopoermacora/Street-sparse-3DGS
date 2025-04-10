#ifndef MESH_LOADER
#define MESH_LOADER

#include <memory>
#include <vector>

#include <cf/3d/io/mesh_file.hpp>
#include <cf/3d/io/openctm_file_url.hpp>
#include <cf/io/url/url_directory_listing.hpp>
#include <cf/container/lru_cache.hpp>
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

#include "render/terrestrial_mesh.hpp"

DECLARE_string(mesh_format);
DECLARE_string(authorization_string);

namespace rdc
{

typedef cf::LruCache<std::string, std::shared_ptr<rdc::TerrestrialMesh> > CacheCell;

class MeshLoader
{
public:

    MeshLoader(RTCDevice device, const std::string& mesh_dir) : device_(device), cache_(30), mesh_dir_(mesh_dir)
    {
        std::vector<std::string> file_names;
        std::string regex_filter =  "[^\\.]+\\.ctm";
        cf::io::GetUrlFileNamesFromUrlDirectory
        (
            mesh_dir_,
            FLAGS_authorization_string,
            regex_filter,
            &file_names
        );
        available_cells_.insert(file_names.begin(), file_names.end());
        for (const auto& m : available_cells_)
        {
            std::cout << m << std::endl;
        }
    }

    std::shared_ptr<rdc::TerrestrialMesh> Load(const std::string& key)
    {
        // first, check the cache
        std::shared_ptr<rdc::TerrestrialMesh> mesh;
        if (cache_.Get(key, &mesh))
        {
            return mesh;
        }

        if (available_cells_.find(key) == available_cells_.end())
        {
            return mesh;
        }

        VLOG(4) << "Loading " << key;

        cv::Point3d offset;
        std::vector<cv::Point3f> vertices;
        std::vector<int> indices;

        bool success = cf::io::ReadOpenCtmFileFromUrl
        (
            mesh_dir_ + "/" + key,
            &offset,
            &vertices,
            &indices,
            FLAGS_authorization_string
        );

        CHECK(success) << "Could not load " << mesh_dir_ << "/" << key;

        mesh = std::shared_ptr<rdc::TerrestrialMesh>(new rdc::TerrestrialMesh
        (
            device_,
            offset,
            vertices,
            indices
        ));

        cache_.Add(key, mesh);

        return mesh;
    }

private:

    RTCDevice device_;
    CacheCell cache_;
    std::string mesh_dir_;
    std::set<std::string> available_cells_;
};

}

#endif
