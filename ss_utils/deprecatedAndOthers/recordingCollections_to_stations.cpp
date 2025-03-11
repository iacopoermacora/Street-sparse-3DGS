#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include "cf/io/recording/recording.hpp"
#include "cf/io/recording/image_representation.hpp"
#include "cf/io/recording/recording_intrinsics.hpp"
#include "cf/io/recording/recording_metadata.hpp"
#include "cf/date_time/gnss_date_time.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

// The collection type needed for render_depth_cyclos
using RecCollection = cf::io::RecordingCollection<cf::io::RecordingIntrinsicsEquirectangular, cf::io::RecordingMetadataTerrestrial, cf::io::ImageRepresentationSkip>;
namespace pt = boost::property_tree;

// Convert degrees to radians
double degToRad(double degrees) {
    return degrees * M_PI / 180.0;
}

void createAndSaveRecordingCollection(const std::string& jsonFilename, const std::string& outputFilename) {
    try {
        // Read JSON file using Boost property_tree
        pt::ptree root;
        pt::read_json(jsonFilename, root);
        
        // Create a new recording collection
        RecCollection collection;
        
        // Get EPSG code from RequestParameters
        int epsgCode = root.get<int>("RequestParameters.ReferenceSystemEpsg");
        
        // Header info
        collection.header.version = cf::io::detail::kLatestStationFileVersion;
        collection.header.recording_source = cf::io::RecordingSource::TERRESTRIAL;
        collection.header.intrinsics_type = cf::io::RecordingIntrinsicsType::EQUIRECTANGULAR;
        collection.header.image_representation_type = cf::io::ImageRepresentationSkip::kType;
        collection.header.epsg_code_xyz = epsgCode; // Use the EPSG code from JSON
        collection.header.epsg_code_z = -1; // Not sure about this one
        
        // Get camera parameters
        int imageHeight = root.get<int>("CameraParameters.2.ImageHeight");
        std::string dcrGeneration = root.get<std::string>("CameraParameters.2.DcrGeneration");
        
        // Intrinsics
        cf::io::RecordingIntrinsicsEquirectangular intrinsic(dcrGeneration, imageHeight);
        collection.intrinsics.push_back(intrinsic);
        
        // Process extrinsics and metadata from RecordingProperties
        for (const auto& item : root.get_child("RecordingProperties")) {
            const auto& recording = item.second;
            
            // Create extrinsic
            cf::io::RecordingExtrinsics extrinsic;
            extrinsic.intrinsics_index = 0; // Referencing the first (and only) intrinsic
            
            // Get ImageId
            std::string imageId = recording.get<std::string>("ImageId");
            extrinsic.id = imageId;
            
            // Position (X, Y, Height)
            double x = recording.get<double>("X");
            double y = recording.get<double>("Y");
            double height = recording.get<double>("Height");
            extrinsic.position = cv::Point3d(x, y, height);
            
            // Position standard deviation
            double xStdDev = recording.get<double>("XStdDev");
            double yStdDev = recording.get<double>("YStdDev");
            double heightStdDev = recording.get<double>("HeightStdDev");
            extrinsic.stddev_position = cv::Point3d(xStdDev, yStdDev, heightStdDev);
            
            // Orientation (Yaw) - assuming this is already in radians based on data values
            double yaw = recording.get<double>("Yaw");
            extrinsic.orientation = cv::Point3d(0.0, 0.0, yaw);
            
            // Orientation standard deviation
            double yawStdDev = recording.get<double>("YawStdDev");
            extrinsic.stddev_orientation = cv::Point3d(0.0, 0.0, yawStdDev);
            
            collection.extrinsics.push_back(extrinsic);
            
            // Create matching metadata for this extrinsic
            std::string recordingTimeStr = recording.get<std::string>("RecordingTimeGps");
            
            // Use the provided function to convert ISO 8601 datetime to GnssDateTime
            cf::GnssDateTime timestamp = cf::GnssDateTimeFromIso8601DateTime(recordingTimeStr);
            
            double groundLevelOffset = recording.get<double>("GroundLevelOffset");
            
            // VehicleDirection - convert to radians if it's in degrees
            double vehicleDirection = recording.get<double>("VehicleDirection");
            vehicleDirection = degToRad(vehicleDirection); // Convert to radians
            
            cf::io::RecordingMetadataTerrestrial metadata(timestamp, groundLevelOffset, vehicleDirection);
            collection.metadata.push_back(metadata);
        }
        
        // Set up metadata layout
        collection.metadata_layout = cf::io::RecordingMetadataTerrestrial::GetLayout();
        
        // Image representation
        cf::io::ImageRepresentationSkip imageRep;
        collection.image_representation = imageRep;
        
        // Save to disk using the Write method
        RecCollection::Write(
            outputFilename,
            collection.header.recording_source,
            collection.header.epsg_code_xyz,
            collection.header.epsg_code_z,
            collection.intrinsics,
            collection.extrinsics,
            collection.metadata_layout,
            collection.metadata,
            collection.image_representation
        );
        
        std::cout << "Successfully converted JSON to stations file: " << outputFilename << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error processing JSON file: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_json_file> <output_stations_file>" << std::endl;
        return 1;
    }
    
    std::string jsonFilename = argv[1];
    std::string outputFilename = argv[2];
    
    createAndSaveRecordingCollection(jsonFilename, outputFilename);
    
    return 0;
}