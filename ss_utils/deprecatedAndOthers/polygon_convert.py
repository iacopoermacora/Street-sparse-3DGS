import json

def get_api_polygon(geojson_data):
    """
    Extracts the EPSG:28992 polygon string from GeoJSON data for an API.

    Args:
        geojson_data (dict): The GeoJSON data as a Python dictionary.

    Returns:
        str: The polygon string formatted for the API, or None if an error occurs.
    """
    try:
        rd_coordinates = geojson_data["features"][0]["geometry"]["coordinates"][0]
        polygon_string = "POLYGON((" + ", ".join(f"{lon} {lat}" for lon, lat in rd_coordinates) + "))"
        return polygon_string
    except (KeyError, IndexError, TypeError):
        return None  # Return None if there's an issue with the GeoJSON structure

# Example usage (replace with your actual JSON input):
geojson_string = """
{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[145600.3860892637,424914.48531534185],[145707.27632414497,424973.078490817],[145788.06467522995,424825.6980876707],[145681.17444034867,424767.1049121956],[145600.3860892637,424914.48531534185]]]},"properties":{"id":"Polygon-feature-zbav1fjblpjz2tkrwy8q3e4d","name":"Rectangle","group":"active-measurements","measureDetails":[{},{},{},{}],"fontsize":15,"dimension":2,"measurementTool":"MAP","customGeometryType":"Rectangle","derivedData":{"unit":"m","precision":1,"coordinateStdevs":[[],[],[],[],[]],"area":{"value":20487.171939849854},"segmentLengths":{"value":[121.89619569632302,168.07064259585948,121.89619569632302,168.07064259585948]},"totalLength":{"value":579.933676584365},"deltaXY":{"value":207.62086462545642}},"measureReliability":"RELIABLE","pointsWithErrors":[],"validGeometry":true,"measurementError":[],"date":1741165936731,"color":[255,128,0],"observationLines":{"activeObservation":-1,"selectedMeasureMethod":null},"wgsGeometry":{"type":"Polygon","coordinates":[[[5.250894236716374,51.81277453379918],[5.252442724111238,51.81330297820413],[5.253618204557364,51.81197960940925],[5.25206974881359,51.8114511804463],[5.250894236716374,51.81277453379918]]]}}}],"crs":{"type":"name","properties":{"name":"EPSG:28992"}}}
"""

geojson_data = json.loads(geojson_string)

polygon_result = get_api_polygon(geojson_data)

if polygon_result:
    print(polygon_result)
else:
    print("Error: Invalid GeoJSON data.")