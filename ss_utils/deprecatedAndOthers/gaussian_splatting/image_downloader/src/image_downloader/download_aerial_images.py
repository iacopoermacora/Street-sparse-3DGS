
try:
    import cf_py_importer
except ModuleNotFoundError:
    from . import cf_py_importer

import argparse
import json
from multiprocessing import Pool
from os import PathLike
from pathlib import Path

import cv2
from shapefile import Reader
from tqdm import tqdm

from cf_image.io import atlas
from cf_image.io.atlas.oblique import DownloadFullImage

from cf.io import GetSecret
from cf.io.recording.aerial_recording_details_client import AerialRecordingDetailsClient
from cf.logging import GetLogger

from cf.io.recording.ogc_filter import OgcFilter

logger = GetLogger()

MODE_OPTIONS = ["details", "images", "images_and_details"]


def _BBoxFromShapefile(shapefile):
    if not isinstance(shapefile, str) and isinstance(shapefile, PathLike):
        shapefile = str(shapefile)
    reader = Reader(shapefile)
    shapes = reader.shapes()
    return shapes[0].bbox


def DownloadImageDetailsFromShapefile(shapefile: Path, epsg_code: int, username: str, password: str, output: Path = None, max_features: int = 10000,
                                      ogc_filter: str = None) -> dict:
    """
    Used to download aerial images within the bounding box.

      Possibly upgrade to be able to provide the bbox in a different way
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    # get aerial recording details from api
    bbox = _BBoxFromShapefile(str(shapefile))
    client = AerialRecordingDetailsClient(username, password)
    details = client.GetFeatureByBoundingBox(bbox=bbox, epsg_code=epsg_code, maxFeatures=max_features, filter=ogc_filter)
    if len(details["features"]) >= max_features:
        logger.warning(
            f"Reached limit of {max_features} aerial recordings, some recordings may not be downloaded, consider using smaller area or larger maxFeatures")
    logger.info(f"Downloaded details of {len(details['features'])} images, saving to {output}")

    # save details to file
    with open(output, "w") as outfile:
        json.dump(details, outfile, indent=4)

    return details


def DownloadImagesFromDetails(details: dict, output_dir: Path, skip_existing=True, zoom_level=0, processes=1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    to_download = details
    if skip_existing:
        to_download = list(filter(lambda image_details: not (output_dir / f"{image_details['id']}.png").exists(), details["features"]))

    # download images
    logger.info(f"Downloading {len(to_download)} images")

    def DownloadImageAndSave(image_details):
        image_array = DownloadFullImage(image_details["id"], zoom_level, image_details['properties']["width"], image_details['properties']["height"])
        cv2.imwrite(str(output_dir / f"{image_details['id']}.jpg"), image_array)

    if processes < 2:
        for image_details in tqdm(to_download):
            DownloadImageAndSave(image_details)
    else:
        with Pool(processes=processes) as pool:
            pool.map(DownloadImageAndSave, to_download)


def _ParseArguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--shapefile', type=str, help='Path to the details file downloaded from the atlas oblique API')
    parser.add_argument('--epsg_code', type=int, help='Code normally found in crs strings like EPSG:_')

    parser.add_argument('--mode', type=str, choices=MODE_OPTIONS, default=MODE_OPTIONS[2], help="What mode to run the script in.")

    parser.add_argument('--output_dir', type=Path, help='Where to save all the images (and by default the details file)')
    parser.add_argument('--details_file', type=Path, default=None, help='Override for where the details should be saved (.json)')

    parser.add_argument('--username', type=str, default=None,
                        help='Username used for authentication with AORS and remote filesystem, used in conjunction with either --password or --password_key')
    parser.add_argument('--password', type=str, default=None, help='Password used for authentication with AORS and remote filesystem')
    parser.add_argument('--password_key', type=str, default=None, help='Password key used to retrieve the password from the keystore.')
    parser.add_argument('--max_recordings', type=int, default=10000,
                        help='Set a limit on how many recording details can be downloaded (does not work with mode=images)')
    parser.add_argument('--year', type=str, default=None, help='Only gets recordings and details from the specified year')
    parser.add_argument('--num_processes', type=int, default=1, help='Use multiple processes to download images in parallel')
    parser.add_argument('--force_redownload_images', action='store_true', help='By default existing images are not re-downloaded, this overrides that')

    args = parser.parse_args()
    if args.mode == MODE_OPTIONS[0] or args.mode == MODE_OPTIONS[1]:
        assert args.details_file is not None, "details_file path must be provided"
    elif args.mode == MODE_OPTIONS[1] or args.mode == MODE_OPTIONS[2]:
        assert args.output_dir is not None, 'output_dir to save the images must be provided'

    assert args.password is None or args.password_key is None, "Cannot provide both password and password_key at the same time"
    return args


def Main(args):
    # Use the cyclodetect user by default
    if not args.username:
        args.username = 'cyclodetect'
        args.password_key = 'atlas-cyclodetect'

    if args.password_key is not None:
        args.password = GetSecret(args.password_key)

    # Downloading images using a bounding box obtained fro a shapefile
    # shapefile = Path("/media/raid_1/jmys/Nijmegen001-003_50x50_0-0/shape/Polygon_2D.shp")
    # epsg_code = 28992

    # username = 'cyclodetect'
    # password_key = 'atlas-cyclodetect'

    # output_dir = Path("/media/raid_1/jmys/Nijmegen001-003_50x50_0-0/")

    if args.mode != MODE_OPTIONS[1]:
        ogc_filter = OgcFilter.Eq("year", args.year) if args.year else None

        details = DownloadImageDetailsFromShapefile(
            args.shapefile,
            args.epsg_code,
            args.username,
            args.password,
            output=args.details_file,  # args.output_dir / "details-2023.json",
            max_features=args.max_recordings,
            ogc_filter=ogc_filter
        )
    else:
        with open(args.details_file, 'r') as f:
            details = json.load(f)

    if args.mode in MODE_OPTIONS[1:]:
        # Make sure the filesystem where the images are stored is accessible
        atlas.filesystem.RegisterFileSystemFromUserNamePassword(
            args.username,
            args.password,
            sub='image/oblique',
        )
        DownloadImagesFromDetails(details, args.output_dir)


if __name__ == '__main__':
    Main(_ParseArguments())
