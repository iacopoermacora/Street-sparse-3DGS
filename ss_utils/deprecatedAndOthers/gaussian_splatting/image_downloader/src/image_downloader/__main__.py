"""
Download color images and distance maps (and recording details)

This script heavily relies on billboards_prepare_data. Therefore, it currently
 - only downloads images in cubic tiling
 - only rectilinear projection
 - it skips the up and down images
 - it skips images already downloaded.

See also https://cyclomedia.atlassian.net/wiki/spaces/AIR/pages/1646034956/How+to+download+all+images+depth+maps+in+an+area+using+airwolf+cityfusion+code+base

EXAMPLES

<PROJECT_ROOT>/src/active/cityfusion/tasks/billboards/billboards_adp/data/test/dev_inventory_project_settings_billboards.xml
/mnt/cm-nas02/tmp/tmp/bvs/billboards_detection/shapes/kinkerstraat/kinkerstraat.shp
<PROJECT_ROOT>/build/tmp/output_download_images
2021-07-01

"""
import argparse
import datetime
import json
import os
from typing import Optional

import cf_py_importer
import cf
import cf.geographic
import cf.inventory.flags
import cf.io.recording.recording_details_client as recording_details_client
import cf.io.recording.recording_details_request_helpers as recording_details_request_helpers
import cf.logging
import cf.subprocess.cityfusion_subprocess as cityfusion_subprocess
import cf_inventory.settings

logger = cf.logging.GetLogger()


def _Main():
    arguments = _ParseArguments()
    _DownloadImagesForRegion(arguments.inventory_project_settings_file_path,
                             arguments.working_dir,
                             arguments.region,
                             arguments.directions,
                             arguments.date_from,
                             date_till=arguments.date_till)
    logger.info('Finished downloading images')


def _DownloadImagesForRegion(inventory_project_settings_file_path: str,
                             working_dir: str,
                             region_file_path: str,
                             directions: str,
                             date_from: datetime.datetime,
                             date_till: Optional[datetime.datetime]):
    recording_details, region = _RetrieveRecordingDetails(inventory_project_settings_file_path, region_file_path, date_from, date_till)
    dataset_names = set(recording['Dataset'] for recording in recording_details['RecordingProperties'])
    if len(dataset_names) > 1:
        logger.warning('Downloading from multiple datasets: {}'.format(', '.join(dataset_names)))

    recording_details_path = os.path.join(working_dir, 'recording_details.pickle')
    cf.io.WritePickle(recording_details_path, recording_details)
    cf.io.WriteString(
        os.path.join(working_dir, 'recording_details.json'),
        json.dumps(recording_details),
    )

    logger.info('Start downloading images')
    cityfusion_subprocess.CheckOutputCityfusionPython(
        'gaussians_prepare_data',
        [
            working_dir,
            '--recording_details_path',
            recording_details_path,
            '--selected_area',
            cf.inventory.flags.EncodeRegionArgument(geometry=region.AsShapelyGeometry(), epsg_code=region.GetEpsgCode()),
            '--directions',
            directions
        ]
    )


def _RetrieveRecordingDetails(inventory_project_settings_file_path, region_file_path, date_from, date_till):
    rds_client = cf_inventory.settings.CreateRecordingDetailsClientFromInventorySettings(inventory_project_settings_file_path)
    region = cf.geographic.LoadProcessingArea(region_file_path)
    recording_details = recording_details_request_helpers.RetrieveLargeArea(rds_client,
                                                                            area=region.AsShapelyGeometry().wkt,
                                                                            areabuffer=0.0,
                                                                            input_reference_system=region.GetEpsgCode(),
                                                                            output_reference_system=region.GetEpsgCode(),
                                                                            recording_type=1,  # No aquaramas
                                                                            )
    recording_details = recording_details_request_helpers.FilterRecordingDetailsByTime(
        recording_details,
        date_from=date_from,
        date_till=date_till,
    )
    return recording_details, region


def _ParseArguments():
    parser = argparse.ArgumentParser('Download images and depth maps')
    parser.add_argument('--inventory_project_settings_file_path',
                        help='Settings file with the RDS parameters AND SAS token credentials')
    parser.add_argument('--region',
                        type=lambda value: cf.ValidateStringIsUrlOrExistingFile('region', value),
                        help='File with geographic region like a SHP file')
    parser.add_argument('--working_dir',
                        type=lambda value: cf.ValidateStringIsUrlOrExistingDirectory('working_dir', value),
                        help='Directory to write the images to. Should exist.')
    parser.add_argument('--date_from', type=cf.ValidateStringIsDate,
                        help='Ignore images recorded before this date')
    parser.add_argument('--date_till', type=cf.ValidateStringIsDateOrEmpty)
    parser.add_argument('--directions', type=str, default='1', choices=['1', '2', '3', '4'], 
                        help='Camera directions: 1=FRLB, 2=F1F2R1R2B1B2L1L2, 3=F1F2R1R2B1B2L1L2U1U2, 4=FRLBU1U2')
    arguments = parser.parse_args()
    if arguments.date_till and arguments.date_till <= arguments.date_from:
        parser.error('Please select a date_till which is later than date_from')
    return arguments


if __name__ == '__main__':
    _Main()
