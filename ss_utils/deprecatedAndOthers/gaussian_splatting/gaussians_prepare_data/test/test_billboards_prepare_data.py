import os

import shapely.geometry

from . import config_test_data  # Break coding style to add to import paths
import cf.geographic

import billboards_prepare_data


_RECORDING_DETAILS_FILE_PATH = os.path.join(config_test_data.GetProjectRootDir(), 'src/active/cityfusion/tools/ml/cdump_processer/data/test/recording_details.pkl')
Y0 = 4509655
Y1 = 4509656
X0 = 584474
X1 = 584475
_PROCESSING_AREA = cf.geographic.ProcessingArea(shapely.geometry.Polygon([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)]), 'EPSG:26918')


def test_LoadRecordingDetails_Plain():
    recording_details = billboards_prepare_data.LoadRecordingDetails(_RECORDING_DETAILS_FILE_PATH, None)
    assert len(recording_details['RecordingProperties']) == 3


def test_LoadRecordingDetails_WithProcessingArea():
    recording_details = billboards_prepare_data.LoadRecordingDetails(_RECORDING_DETAILS_FILE_PATH, _PROCESSING_AREA)
    assert len(recording_details['RecordingProperties']) == 1

