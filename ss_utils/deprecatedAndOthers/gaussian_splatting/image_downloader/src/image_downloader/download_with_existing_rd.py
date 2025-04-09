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
    _DownloadImagesForRegion(arguments.recording_details_file_path,
                             arguments.working_dir,
                             arguments.directions)
    logger.info('Finished downloading images')


def _DownloadImagesForRegion(recording_details_file_path: str,
                             working_dir: str,
                             directions: str):
    with open(recording_details_file_path, 'r') as f:
            recording_details = json.load(f)
    dataset_names = set(recording['Dataset'] for recording in recording_details['RecordingProperties'])
    if len(dataset_names) > 1:
        logger.warning('Downloading from multiple datasets: {}'.format(', '.join(dataset_names)))

    recording_details_path = os.path.join(working_dir, 'recording_details.pickle')
    # Write recording details as JSON as well.
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
            '--directions',
            directions
        ],
        should_search_experimental=True
    )


def _ParseArguments():
    parser = argparse.ArgumentParser('Download images and depth maps')
    parser.add_argument('--recording_details_file_path',
                        help='path to the recording details file')
    parser.add_argument('--working_dir',
                        type=lambda value: cf.ValidateStringIsUrlOrExistingDirectory('working_dir', value),
                        help='Directory to write the images to. Should exist.'),
    parser.add_argument('--directions', type=str, default='1', choices=['1', '2', '3', '4'], 
                        help='Camera directions: 1=FRLB, 2=F1F2R1R2B1B2L1L2, 3=F1F2R1R2B1B2L1L2U1U2, 4=FRLBU1U2')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    _Main()
