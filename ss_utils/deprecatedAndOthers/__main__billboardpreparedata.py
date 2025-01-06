"""
Prepare for a billboards project by downloading recording details and images

EXAMPLE

The inventory project settings file defines the SAS token.

file:///mnt/cm-nas02/tmp/dev_billboards/adp_tests/BILLBOARDS_NL-ABCD-20190807-testing-veemarkt-small_20191114T223135/inventory_project_settings.xml
file:///mnt/cm-nas02/tmp/dev_billboards/adp_tests/BILLBOARDS_NL-ABCD-20190807-testing-veemarkt-small_20191114T223135
--recording_details_path
file:///mnt/cm-nas02/tmp/dev_billboards/adp_tests/BILLBOARDS_NL-ABCD-20190807-testing-veemarkt-small_20191114T223135/recording_details.pkl
--selected_area
base64U1JJRD0yODk5MjtQT0xZR09OICgoMTI0MjUwLjAwMDAwMDAwMDAwMDAwMDAgNDg2NTAwLjAwMDAwMDAwMDAwMDAwMDAsIDEyNDUwMC4wMDAwMDAwMDAwMDAwMDAwIDQ4NjUwMC4wMDAwMDAwMDAwMDAwMDAwLCAxMjQ1MDAuMDAwMDAwMDAwMDAwMDAwMCA0ODY3NTAuMDAwMDAwMDAwMDAwMDAwMCwgMTI0MjUwLjAwMDAwMDAwMDAwMDAwMDAgNDg2NzUwLjAwMDAwMDAwMDAwMDAwMDAsIDEyNDI1MC4wMDAwMDAwMDAwMDAwMDAwIDQ4NjUwMC4wMDAwMDAwMDAwMDAwMDAwKSk=

"""
import argparse
import copy
import logging
import math
import os

import numpy as np
import shapely.geometry

try:
    import cf_py_importer
except ModuleNotFoundError:
    from . import cf_py_importer
import cf.geographic
import cf.image
import cf.io
from cf.io.filesystem.io_image import WriteImage
import cf.logging
import cf.inventory.flags
from cf.io.image.cubic_image_downloader import (
    CreateCubicImageDownloader,
    DownloadImages
)

from cf.io.image.depth_map_to_distances import (
    ConvertDepthMapToDistanceImageInMeters,
    VisualizeDistance
)
from cf.io.recording.recording_details_parsers import ParsePoseCorrectionsFromRecordingDetails
import cf.validate_flags

from cf_image.remap import (
    RemapColor,
    RemapDepth,
    StackCubicFaceImagesVertically,
    ComputeRemapTablesCubicToRectilinear
)
import cf_inventory.settings


cf.logging.SetBasicConfigIfMain(__name__, logger_name='BILLBOARDS_PREPARE_DATA')


def _ParseArgs():
    parser = argparse.ArgumentParser( 'Prepare billboard detections by downloading images' )
    parser.add_argument('working_dir', metavar='WORKING_DIR',
                        type=str, # if on the blob, the container must exist already
                        help='Path to working directory. Images will be stored in a sub-directory of this.')
    parser.add_argument('--recording_details_path',
                        type=lambda value: cf.validate_flags.ValidateStringIsUrlOrExistingFile('recording_details_path', value),
                        help='Add the recording details file path to avoid fetching new recording details from the remote service. '
                             'However, the caller needs to make sure that the output EPSG code matches the RDS results.')
    parser.add_argument('--selected_area', type=_DecodeProcessingAreaFromCommandLineArgument,
                        help='EWKT region description base64 encoded area')
    cf.io.AddFileSystemConfigFlag(parser)  # https://cyclomedia.atlassian.net/wiki/spaces/AIR/pages/399671359/cf+io+and+the+virtual+filesystem
    args = parser.parse_args()
    return args


def _DecodeProcessingAreaFromCommandLineArgument(value):
    region_descriptor = cf.inventory.flags.DecodeRegionArgument(value)
    if 'geometry' not in region_descriptor:
        raise NotImplementedError('Does not support other region descriptors but geometries')
    if len(region_descriptor['geometry']) != 1:
        raise NotImplementedError('Did not implemented combining several regions into one')
    return region_descriptor['geometry'][0].AsProcessingArea()


def CheckExpectedOutputAlreadyExist(
    image_id,
    face_names,
    path_templates
):
    paths = []
    for face_name in face_names:
        for path_template in path_templates:
            paths.append( path_template.format( image_id, face_name ) )
    result = all( [ cf.io.FileExists( path ) for path in paths ] )
    return result


def LoadRecordingDetails(recording_details_path, selected_area):
    recording_details = cf.io.ReadPickle(recording_details_path)
    if selected_area:
        recording_details = _FilterRecordingsByProcessingArea(recording_details, selected_area)
    return recording_details


def PrepareData(
    recording_details,
    download_image_zoom_level,
    cubic_image_downloader,
    out_color_image_path_template,
    out_distance_image_path_template,
    out_distance_visualization_path_template
):
    hfov = math.radians(120)  # Changed from 90 to 120 # PACOMMENT: Modified this line

    image_corrections = ParsePoseCorrectionsFromRecordingDetails(recording_details)

    image_ids = [str(recording_properties['ImageId']) for recording_properties in recording_details['RecordingProperties'] ]
    face_names = [ 'f', 'r', 'b', 'l', 'u', 'd'] # PACOMMENT: Added u, d
    
    YAW_PITCH_LUT = { # PACOMMENT: Added this.
        'f': (0, 0),
        'r': (270, 0),
        'b': (180, 0),
        'l': (90, 0),
        'u': (0, -90),
        'd': (0, 90)
    }

    for image_id in image_ids:
        if not image_id in image_corrections:
            continue
        image_correction = image_corrections[image_id]

        # skip if images already exist
        expected_output_already_exists = CheckExpectedOutputAlreadyExist(
            image_id,
            face_names,
            [
                out_color_image_path_template,
                out_distance_image_path_template,
                #out_distance_visualization_path_template
            ]
        )
        if expected_output_already_exists:
           continue

        # apparently the output data doesn't exist yet
        # # first download the images
        color_images, depth_images = DownloadImages(
            image_id,
            download_image_zoom_level,
            cubic_image_downloader
        )

        image_size = color_images.front.rows

        # process the
        for face_name in face_names:
            # YAW_LUT = {'f': 0, 'r': 270, 'b': 180, 'l': 90} # PACOMMENT: Commented
            # yaw_degrees = YAW_LUT[face_name] # PACOMMENT: Commented
            yaw_degrees, pitch_degrees = YAW_PITCH_LUT[face_name] # PACOMMENT: Added
            total_rotations = image_correction["ImageCorrectionForVehicleDirection"] + math.radians(yaw_degrees) - image_correction["ImageCorrectionForNorth"]
            pitch = math.radians(pitch_degrees) # PACOMMENT: Added
            
            map_x, map_y = ComputeRemapTablesCubicToRectilinear(image_size, image_size, hfov, total_rotations, pitch) # PACOMMENT: Added pitch

            color_stack = StackCubicFaceImagesVertically(cf.image.cubic_image_data.ConvertCubicImagesToFaceDict(color_images))
            depth_stack = StackCubicFaceImagesVertically(cf.image.cubic_image_data.ConvertCubicImagesToFaceDict(depth_images))
            distance_stack = ConvertDepthMapToDistanceImageInMeters(depth_stack)

            out_color_image_path    = out_color_image_path_template.format( image_id, face_name )
            out_distance_image_path = out_distance_image_path_template.format( image_id, face_name )
            # out_distance_visualization_path = out_distance_visualization_path_template.format( image_id, face_name )

            remapped_color = RemapColor(color_stack, map_x, map_y)

            cf.io.MakeDirs( os.path.dirname( out_color_image_path ) )
            WriteImage( out_color_image_path, remapped_color )
            logging.info( "Wrote: {}".format(out_color_image_path) )

            remapped_depth = RemapDepth(distance_stack, map_x, map_y)
            cf.io.MakeDirs( os.path.dirname( out_distance_image_path ) )
            cf.io.WriteNumpyFile( out_distance_image_path, remapped_depth )
            logging.info( "Wrote: {}".format(out_distance_image_path) )

            #
            # depth_vis = VisualizeDistance(remapped_depth)
            # depth_vis_path = g_output_path_template_depth_vis.format(image_id, yaw)
            # cv2.imwrite(depth_vis_path, depth_vis)
            # logging.info("Wrote: {}".format(depth_vis_path))

    return


def _FilterRecordingsByProcessingArea(recording_details, processing_area):
    recording_details = copy.deepcopy(recording_details)
    get_point = lambda rec: shapely.geometry.Point(rec['X'], rec['Y'])
    get_epsg_code = lambda rec: recording_details['RequestParameters']['ReferenceSystemEpsg']
    recording_details['RecordingProperties'] = processing_area.FilterObjects(recording_details['RecordingProperties'], get_point, get_epsg_code=get_epsg_code)
    return recording_details


if __name__ == "__main__":
    logging.basicConfig( level=logging.INFO )
    args = _ParseArgs()
    download_image_zoom_level = 2

    cf.io.MakeDirs( args.working_dir )

    recording_details_path = args.recording_details_path or os.path.join(args.working_dir, 'recording_details.pkl')
    out_color_image_path_template = os.path.join(args.working_dir, 'images/level_2/color/{}_{}.jpg')
    out_distance_image_path_template = os.path.join(args.working_dir, 'images/level_2/depth/{}_{}.png.npy')
    out_distance_visualization_path_template = os.path.join(args.working_dir, 'debug/visualization/projected/{}_{}.laz')

    recording_details = LoadRecordingDetails(args.recording_details_path, args.selected_area)

    # Using managed identities. Make sure the permissions are correctly set-up
    # Refer: https://cyclomedia.atlassian.net/wiki/spaces/AIR/pages/1623097442/How+to+support+Managed+Identities+in+our+products
    cubic_image_downloader = CreateCubicImageDownloader(recording_details)

    PrepareData(recording_details,
                download_image_zoom_level,
                cubic_image_downloader,
                out_color_image_path_template,
                out_distance_image_path_template,
                out_distance_visualization_path_template
                )
