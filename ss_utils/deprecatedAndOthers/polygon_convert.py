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
{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[145607.8045651314,424899.5606221837],[145683.4163292941,424941.0081255079],[145730.6389470164,424854.86094780127],[145655.02718285372,424813.4134444771],[145607.8045651314,424899.5606221837]]]},"properties":{"id":"Polygon-feature-zbav1fjblpjz2tkrwy8q3e4d","name":"Rectangle","group":"active-measurements","measureDetails":[{},{},{},{}],"fontsize":15,"dimension":2,"measurementTool":"MAP","customGeometryType":"Rectangle","derivedData":{"unit":"m","precision":1,"coordinateStdevs":[[],[],[],[],[]],"area":{"value":8470.999691009521},"segmentLengths":{"value":[86.22664560101639,98.24109044269123,86.22664560101639,98.24109044269123]},"totalLength":{"value":368.93547208741523},"deltaXY":{"value":130.714751512498}},"measureReliability":"RELIABLE","pointsWithErrors":[],"validGeometry":true,"measurementError":[],"date":1742468380062,"color":[255,128,0],"observationLines":{"activeObservation":-1,"selectedMeasureMethod":null},"wgsGeometry":{"type":"Polygon","coordinates":[[[5.2510022199794815,51.812640510200545],[5.252097579250033,51.813014321081084],[5.2527846840472385,51.8122407855544],[5.25168933786353,51.811866981058834],[5.2510022199794815,51.812640510200545]]]}}}],"crs":{"type":"name","properties":{"name":"EPSG:28992"}}}
"""

geojson_data = json.loads(geojson_string)

polygon_result = get_api_polygon(geojson_data)

if polygon_result:
    print(polygon_result)
else:
    print("Error: Invalid GeoJSON data.")