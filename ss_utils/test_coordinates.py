from exif import Image
import os

def image_coordinates():
    image_path = "/home/local/CYCLOMEDIA001/iermacora/Street-sparse-3DGS/inputsB/images/cam1"
    with open(os.path.join(image_path, "0000_WE92P1B9_f.jpg"), 'rb') as src:
        img = Image(src)
    if img.has_exif:
        try:
            img.gps_longitude
            #coords = [
            #    decimal_coords(img.gps_latitude, img.gps_latitude_ref),
            #    decimal_coords(img.gps_longitude, img.gps_longitude_ref)
            #]
            print(img.gps_longitude)
            return img.gps_longitude
        except AttributeError:
            print('boh')
            return None
    else:
        print('nada')
        return None

image_coordinates()