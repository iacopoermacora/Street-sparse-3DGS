#include "_version_src/version.hpp"

#include <pmmintrin.h>
#include <string>
#include <utility>
#include <xmmintrin.h>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/unordered_map.hpp>
#include <boost/regex.hpp>
#include <boost/filesystem/path.hpp>
#include <cf/geometry/geometry.hpp>
#include <cf/helper_string.hpp>
#include <cf/io/flags.hpp>
#include <cf/io/pano_formats.hpp>
#include <cf/3d/io/mesh_file.hpp>
#include <cf/io/recording/helper_camera.hpp>
#include <cf/io/recording/recording.hpp>
#include <cf/io/recording/image_tile_url_generator.hpp>
#include <cf/io/url/url_reader.hpp>
#include <cf/io/url/url_util.hpp>
#include <cf/io/url/url_writer.hpp>
#include <cf/stdx/memory.hpp>

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tbb/tbb.h>


#include "mesh_loader.hpp"
#include "render/render_meshes.hpp"
#include "render/terrestrial_mesh.hpp"

DEFINE_string
(
    in_terrestrial_stations_file_url,
    "",
    "The input .stations file url for terrestrial recording-locations."
);

DEFINE_string
(
    in_mesh_directory,
    "",
    "Directory with mesh-cells (see also 'in_cell_prefix' parameter)."
);

DEFINE_string
(
    in_cell_prefix,
    "pmg__hires_",
    "Only cells with this prefix get processed."
);

DEFINE_string
(
    out_depth_cyclo_directory_url,
    "",
    "The url of the directory where output depth-cyclorama's are stored. "
    "May also be a url to the ProcessingInfoService: https://[p|d]wecmhosprocessinginfo.azurewebsites.net/api/processinginfo "
    "In that case, the service will be used to look up the location of the depth tiles."
);

DEFINE_string
(
    out_quality_control_directory_url,
    "",
    "Optional. The url of the directory where QC results are stored. A csv file with the fill percentage of each pano will be written here."
);

DEFINE_string
(
    directions,
    "",
    "The identifier of the directions the cameras must be positioned. "
    "1=F1R1B1L1"
    "2=F1F2R1R2B1B2L1L2"
    "3=F1F2R1R2B1B2L1L2U1U2"
    "4=F1R1B1L1U1U2"
);

DEFINE_string
(
    authorization_string,
    "",
    "May be username:password or something else, depending on context."
);

DEFINE_string
(
    proc_info_service_key,
    "",
    "The contents of the x-api-key header for the proc info service if needed for depth cyclo output."
);

DEFINE_double
(
    cell_size,
    50.,
    "Size of (side of) cell in units (meter)."
);

DEFINE_int32
(
    pixels_per_tile_side,
    512,
    "Width and height of a single tile in pixels."
);

DEFINE_int32
(
    tiles_per_face_side,
    3,
    "Width and height of a single face in tiles."
);

namespace rdc
{

void LoadMeshesForStationPosition
(
	const std::string& cell_prefix,
    const cv::Point3d& rec_position,
    rdc::MeshLoader* loader,
    std::vector<std::shared_ptr<rdc::TerrestrialMesh> >* meshes_for_station
)
{
    cv::Point2i cell
        (
            static_cast<int>(rec_position.x / static_cast<double>(FLAGS_cell_size)),
            static_cast<int>(rec_position.y / static_cast<double>(FLAGS_cell_size))
        );

    int boundary = 2;
    for (int y = -boundary; y <= boundary; ++y)
    {
        for (int x = -boundary; x <= boundary; ++x)
        {
            cv::Point2i cell_to_load(cell.x + x, cell.y + y);

            std::string file_name =
                cell_prefix +
                boost::lexical_cast<std::string>(cell_to_load.x) +
                "_" +
                boost::lexical_cast<std::string>(cell_to_load.y) +
                ".ctm";
            std::shared_ptr<rdc::TerrestrialMesh> p_mesh = loader->Load(file_name);
            if (p_mesh)
            {
                meshes_for_station->push_back(p_mesh);
            }
        }
    }
}

cv::Mat DistanceToColor(const cv::Mat& dist_img)
{
    cv::Mat color(dist_img.rows, dist_img.cols, CV_8UC3);

    for (int y = 0; y < dist_img.rows; ++y)
    {
        const float* dist = dist_img.ptr<float>(y);
        cv::Vec3b* col = color.ptr<cv::Vec3b>(y);

        for (int x = 0; x < dist_img.cols; ++x)
        {
            double d = 1000.0 * static_cast<double>(dist[x]); // d is now distance in mm

            uint32_t val = 0;
            if (d < 1.0)                                                    // special values
                val = 0;
            //else if (d < 16384)                                             // 16384 1 mm res
            //    val = static_cast<uint32_t>(d) << 8;
            else if (d <= 65536)                                            // 65536 4mm res
                val = (0x40 << 16) | (static_cast<uint32_t>(d / 4) << 8);
            else if (d <= 262144)                                           // 262144 16mm res
                val = (0x80 << 16) | (static_cast<uint32_t>(d / 16) << 8);
            else if (d <= 1048576)                                          // 1048576 64mm res
                val = (0xC0 << 16) | (static_cast<uint32_t>(d / 64) << 8);

            cv::Vec3b& c = col[x];
            c[2] = 0xFF & (val >> 16); // R
            c[1] = 0xFF & (val >> 8);  // G
            c[0] = 0xFF & (val >> 0);  // B
        }
    }

    return color;
}

double ComputeFillPct(const cv::Mat& depth_image)
{
    int filled_count = 0;

    for (int y = 0; y < depth_image.rows; ++y)
    {
        const cv::Vec3b *col = depth_image.ptr<cv::Vec3b>(y);

        for (int x = 0; x < depth_image.cols; ++x)
        {
            if (col[x] != cv::Vec3b(0, 0, 0))
            {
                filled_count++;
            }
        }
    }

    return 100.0 * static_cast<double>(filled_count) / static_cast<double>(depth_image.rows * depth_image.cols);
}

void RenderDepthForStations
(
    const std::string& in_terrestrial_stations_file_url,
    const std::string& in_mesh_dir,
	const std::string& cell_prefix,
	const std::string& out_cyclo_dir_url,
    const std::string& out_quality_control_dir_url,
    const std::string& directions
)
{
    RTCDevice device = rtcNewDevice(0);

    using PanoRecordings = cf::io::RecordingCollection<cf::io::RecordingIntrinsicsEquirectangular, cf::io::RecordingMetadataTerrestrial, cf::io::ImageRepresentationSkip>;
    PanoRecordings stations_file;

    try
    {
        cf::io::UrlReader reader(in_terrestrial_stations_file_url, FLAGS_authorization_string);
        stations_file = PanoRecordings::Read(&reader.GetIstream());
    }
    catch(std::exception& e)
    {
        LOG(FATAL) << "Could not read " << in_terrestrial_stations_file_url << ": " << e.what();
    }
    const auto& station_extrinsics = stations_file.extrinsics;
    const auto& station_metadata = stations_file.metadata;
    // sort stations on time
    std::vector<std::pair<std::pair<size_t, cf::GnssDateTime>, size_t> > ordened;
    ordened.reserve(station_extrinsics.size());
    for (size_t station_index = 0; station_index < station_extrinsics.size(); ++station_index)
    {
        ordened.push_back
        (
            std::make_pair
            (
                std::make_pair
                (
                    station_extrinsics[station_index].intrinsics_index,
                    station_metadata[station_index].date_time
                ),
                station_index
            )
        );
    }
    std::sort(ordened.begin(), ordened.end());

    cf::io::UrlWriter qc_log(FLAGS_authorization_string, 5000, 5);

    if (!out_quality_control_dir_url.empty())
    {
        cf::io::CreateDirectoryFromUrl(out_quality_control_dir_url);
        std::string write_url = out_quality_control_dir_url;
        cf::AddTrailingSlash(&write_url);
        std::string stations_file_name = in_terrestrial_stations_file_url.substr(1 + in_terrestrial_stations_file_url.find_last_of('/'));
        auto dot_loc = stations_file_name.find_last_of('.');
        CHECK_NE(dot_loc, std::string::npos);
        std::string job_name = stations_file_name.substr(0, dot_loc);
        std::string qc_log_url = write_url + job_name  + "_fill_pct.csv";
        qc_log.Open(qc_log_url);
    }


    bool use_proc_info_service = out_cyclo_dir_url.find("/api/processinginfo") != std::string::npos;
    std::string tile_write_dir;
    std::string dataset;

    if (use_proc_info_service)
    {
        // find the dataset

        if (ordened.empty())
            return;

        tile_write_dir = out_cyclo_dir_url;
        const auto& rec_extr = station_extrinsics[ordened[0].second];
        dataset = cf::io::GetDatasetAndTileSchemeForRecording(out_cyclo_dir_url, rec_extr.id, FLAGS_proc_info_service_key).first;
    }
    else
    {
        // get the last part of the input url as the dataset
        cf::io::Uri uri = cf::io::Uri::FromUriString(out_cyclo_dir_url);
        auto path = uri.path();
        cf::RemoveTrailingSlash(&path);
        auto slash_pos = path.find_last_of("/");
        CHECK_NE(slash_pos, std::string::npos);
        uri.path(path.substr(0, slash_pos));
        dataset = path.substr(slash_pos + 1);
        tile_write_dir = uri.ToString();
    }

    // all device related memory should be released before rtcDeleteDevice(device);
    {
        // Create the cube distance based on the directions argument
        int cube_distance_size = 0;
        if (directions == "1")
            cube_distance_size = 4;
        else if (directions == "2")
            cube_distance_size = 8;
        else if (directions == "3")
            cube_distance_size = 10;
        else if (directions == "4")
            cube_distance_size = 6;
        else
            LOG(FATAL) << "Invalid directions argument: " << directions;
        // Create the cube distance matrix
        cv::Mat cube_distance[cube_distance_size];
        TerrestrialMeshesRenderer renderer(device);
        rdc::MeshLoader loader(device, FLAGS_in_mesh_directory);

        // Create a UrlWriter for each thread, so tcp connections etc. can be saved.
        // The currently used version of tbb does not yet support an Args&&... constructor for
        // enumerable_thread_specific, so we must wrap the object to support a default constructor
        // with the correct arguments.
        struct UrlWriterWrapper
        {
            UrlWriterWrapper() : writer(FLAGS_authorization_string, 10000)
            {
            }

            cf::io::UrlWriter writer;
            std::unique_ptr<cf::io::ImageTileUrlGenerator> url_generator;

        };

        typedef tbb::enumerable_thread_specific<UrlWriterWrapper> ThreadSpecificUrlWriter;
        ThreadSpecificUrlWriter url_writers;

        // Create the face names based on the directions argument
        std::vector<char> face_names;
        if (directions == "1") {
            face_names = {'F', 'R', 'B', 'L'};
        } else if (directions == "2") {
            face_names = {'F', 'F', 'R', 'R', 'B', 'B', 'L', 'L'};
        } else if (directions == "3") {
            face_names = {'F', 'F', 'R', 'R', 'B', 'B', 'L', 'L', 'U', 'U'};
        } else if (directions == "4") {
            face_names = {'F', 'R', 'B', 'L', 'U', 'U'};
        } else {
            LOG(FATAL) << "Invalid directions argument: " << directions;
        }

        // render the depth maps for each station
        for (size_t i = 0; i < ordened.size(); ++i)
        {
            const auto& rec_extr = station_extrinsics[ordened[i].second];
            const auto& rec_metadata = station_metadata[ordened[i].second];
            float driving_direction = rec_metadata.driving_direction; // Get driving direction

            std::vector<std::shared_ptr<rdc::TerrestrialMesh> > meshes_for_station;
            LoadMeshesForStationPosition(cell_prefix, rec_extr.position, &loader, &meshes_for_station);

            if (meshes_for_station.empty())
            {

                if (!out_quality_control_dir_url.empty())
                {
                    qc_log.GetOstream() << rec_extr.id << ",0.0,false" << std::endl;
                }
                VLOG(4) << "Mesh neighborhood void, skip empty depth-cyclo, continue/next.";
                continue;
            }

            VLOG(4) << "Will render " << meshes_for_station.size() << " meshes.";

            renderer.Render(rec_extr, meshes_for_station, cube_distance, driving_direction, directions);

            VLOG(4) << "Rendered " << (i+1) << " of " << ordened.size();

            cv::Mat depths_as_color[cube_distance_size];
            std::string faces_names[cube_distance_size];

            for (int cube_face = 0; cube_face < cube_distance_size; ++cube_face)
            {
                depths_as_color[cube_face] = DistanceToColor(cube_distance[cube_face]);
            }


            // handle the zoom 1 lod

            tbb::blocked_range<int> faces_range(0, cube_distance_size);
            tbb::parallel_for
            (
                faces_range,
                [&](const tbb::blocked_range<int>& r)
                {
                    for (int cube_face = r.begin(); cube_face != r.end(); ++cube_face)
                    {
                        cv::Mat& depth_as_color = depths_as_color[cube_face];
                        cv::Mat resized;
                        cv::Size face_size(depth_as_color.cols / 3, depth_as_color.rows / 3);
                        cv::resize(depth_as_color, resized, face_size, 0.0, 0.0, cv::INTER_NEAREST);
                        depth_as_color = resized;

                        int params[] = {cv::IMWRITE_PNG_COMPRESSION, 9, cv::IMWRITE_PNG_STRATEGY, cv::IMWRITE_PNG_STRATEGY_FILTERED};
                        std::vector<int> params_vec(params, params + 4);
                        std::vector<uint8_t> compressed_image;
                        cv::imencode(".png", depth_as_color, compressed_image, params_vec);

                        cf::io::UrlWriter& local_writer = url_writers.local().writer;
                        auto& url_generator = url_writers.local().url_generator;
                        if (!url_generator)
                        {
                            if (use_proc_info_service)
                            {
                                url_generator = cf::stdx::make_unique<cf::io::ProcessingInfoServiceImageTileUrlGenerator>(tile_write_dir, dataset, cf::io::kDcr9TileScheme, true, FLAGS_proc_info_service_key);
                            }
                            else
                            {
                                url_generator = cf::stdx::make_unique<cf::io::SimpleImageTileUrlGenerator>(tile_write_dir, dataset, cf::io::kDcr9TileScheme, true);
                            }
                        }
                        int face_number;
                        if (directions == "1"){
                            face_number = 1;
                        } else if (directions == "2" || directions == "3") {
                            face_number = cube_face % 2;
                        } else if (directions == "4") {
                            face_number = (cube_face != 5) ? 1 : 2;
                        }
                        // TODO: Set face numbers accordingly
                        std::string tile_name = url_generator->GenerateTileUrl(rec_extr.id, face_number, face_names[cube_face], 0, 0);

                        try
                        {
                            local_writer.Open(tile_name);

                            local_writer.WriteBytes(&compressed_image[0], compressed_image.size(), 1);

                            local_writer.Close();
                        }
                        catch (std::exception& ex)
                        {
                            LOG(FATAL) << "Could not write " << tile_name << ": " << ex.what();
                        }
                    }
                }
            );

            VLOG(4) << "Written " << (i + 1) << " of " << ordened.size();
        }

    } // all device related memory should be released before rtcDeleteDevice(device);

    if (!out_quality_control_dir_url.empty())
    {
        try
        {
            qc_log.Close();
        }
        catch (std::exception& e)
        {
            LOG(FATAL) << e.what();
        }
    }


    VLOG(4) << "Done generating depth cycloramas";

    rtcDeleteDevice(device);
}

} // namespace rdc

int main(int argc, char** argv)
{
    google::SetUsageMessage("./render_depth_cyclos --in_terrestrial_stations_file_url <url.stations-file> --in_mesh_directory <dir> --out_depth_cyclo_directory_url <url dir>\n");
    google::SetVersionString(version::GetBuildInfo(true));
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    tbb::task_scheduler_init init;

    bool bail_early = false;

    if (FLAGS_in_terrestrial_stations_file_url == "")
    {
    	LOG(ERROR) << "Need, but missing: --in_terrestrial_stations_file_url";
    	bail_early = true;
    }

    if (FLAGS_in_mesh_directory == "")
    {
    	LOG(ERROR) << "Need, but missing: --in_mesh_directory";
    	bail_early = true;
    }

    if (FLAGS_out_depth_cyclo_directory_url == "")
    {
    	LOG(ERROR) << "Need, but missing: --out_depth_cyclo_directory_url";
    	bail_early = true;
    }

    if (FLAGS_directions == "")
    {
    	LOG(ERROR) << "Need, but missing: --directions";
    	bail_early = true;
    }

    if (bail_early)
    {
    	LOG(FATAL) << "Missing parameters, early exit.";
    }

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    cf::io::StartAllIoFromCommandLineFlags();

    rdc::RenderDepthForStations
    (
        FLAGS_in_terrestrial_stations_file_url,
        FLAGS_in_mesh_directory,
        FLAGS_in_cell_prefix,
        FLAGS_out_depth_cyclo_directory_url,
        FLAGS_out_quality_control_directory_url,
        FLAGS_directions
    );

    cf::io::StopAllIo();

    VLOG(5) << "render_depth_cyclos done";
    return 0;
}
