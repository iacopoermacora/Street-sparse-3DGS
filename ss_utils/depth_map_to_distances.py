# Modified Cyclomedia's code
# projects/cityfusion_master/src/active/cityfusion/libs/cf_py/src/cf/io/image/depth_map_to_distances.py

import cv2
import numpy as np
import os


def ConvertDepthMapToDistanceImageInMeters( depth_image ):

    """
    Takes a depth map and converts the encoded depth to actual floating point distances in meters.
    See: http://wiki/bloodhound/wiki/BasicDepthCycloramasVisualizingAndOutputFormat

    :param depth_image: The depth map to convert
    :return: An image with the actual floating point distances in meters
    """

    [ _, g, r ]   = cv2.split( depth_image )
    r = r.astype( np.int64 )
    g = g.astype( np.int64 )

    precision       = np.left_shift( np.right_shift( r, 6 ), 1 )
    units           = np.bitwise_or( np.left_shift( np.bitwise_and( r, 63 ), 8 ), g )
    depth_in_mm     = np.left_shift( units, precision )
    depth_in_mm_f   = depth_in_mm.astype( np.double )
    depth_in_m_f    = np.divide( depth_in_mm_f, 1000 )
    return depth_in_m_f

if __name__ == "__main__":
    # Import all the depth maps from the folder /home/local/CYCLOMEDIA001/iermacora/testMeshes
    for file in os.listdir("ss_utils/testMeshes"):
        if file.endswith(".png"):
            depth_map = cv2.imread(os.path.join("ss_utils/testMeshes", file), cv2.IMREAD_UNCHANGED)
            distance_image = ConvertDepthMapToDistanceImageInMeters(depth_map)

            # Save the distance image as {file}_bw.png
            cv2.imwrite(os.path.join("ss_utils/testMeshes", file)[:-4] + "_bw.png", distance_image)