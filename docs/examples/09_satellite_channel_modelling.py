#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Satellite Channel Modelling
======================================================

This example uses the frequency domain :func:`lyceanem.models.frequency_domain.calculate_scattering` function to
predict the scattering parameters for the frequency and environment included in the model. This allows the effects of antenna polarisation,


"""

# %%
# Define Map and Ground Station Location and list of Satellites using skyfield API to load starlink satellites.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import folium

interest_center = [51.458729, -2.602153]  # location of Bristol Physics Observatory

m = folium.Map(
    location=[
        interest_center[0],
        interest_center[1],
    ],
    zoom_start=6,
    tiles="Stamen Terrain",
)

import numpy as np
import copy

from skyfield.api import load, EarthSatellite, wgs84

BristolPhysicsObservatory = wgs84.latlon(interest_center[0], interest_center[1], 0)

receiver = BristolPhysicsObservatory

# Build the time range `t` over which to plot, plus other values.

ts = load.timescale()

time_points = 2 * 60  # track over 10 minutes
# change over time
t = ts.utc(2022, 12, 15, 13, 4, range(0, time_points))
timebase = t.utc_datetime()

# %%
# GeoJSON
# ~~~~~~~~
# The Folium plugin `TimestampedGeoJson`_ will be used to plot our tracks using timestamped
# GeoJSONs. As a result, we want to convert our data into `GeoJSON format`_. Firstly, we create
# our feature collection which we will append our features to.
#
# .. _GeoJSON format: https://geojson.org/
# .. _TimestampedGeoJson:  https://python-visualization.github.io/folium/plugins.html

geo_features = list()
geo_json = {
    "type": "FeatureCollection",
    "features": geo_features,
}
# %%
# Plotting Tracks of Satellites visible to Physics Observatory
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now we append our tracks to our feature collection list. We define `colour_iter` which will
# allow us to cycle through different colours when mapping our tracks.

from collections import defaultdict
from itertools import cycle

colour_iter = iter(
    cycle(
        [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "darkred",
            "#0909FF",
            "#F70D1A",
            "#FF6700",
            "lightgreen",
            "#0AFFFF",
            "#12AD2B",
            "#E2F516",
            "#FFFF00",
            "#F52887",
        ]
    )
)
colour = defaultdict(lambda: next(colour_iter))

trail_Size = 7  # trail_Size is the number of timestamps we want track to trail for
from io import BytesIO
import base64


# %%
# Define Starlinks Tracks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Convinience function to limit the tracks to a set number of the nearest N satellites.

def starlink_tracks(pointofinterest, timebase, num_satellites=5):
    starlink_url = "https://celestrak.com/NORAD/elements/supplemental/starlink.txt"
    starlinks = load.tle_file(starlink_url)
    tracks = []
    for sat in starlinks:
        pos = sat.at(timebase)
        tracks.append(pos)
        latcheck = np.any(np.isnan(wgs84.subpoint_of(pos).latitude.degrees))
        loncheck = np.any(np.isnan(wgs84.subpoint_of(pos).longitude.degrees))
        if latcheck | loncheck:
            tracks.pop(-1)
            print("caught one")

    distance_trim = np.full((len(tracks), len(timebase)), np.inf)
    for track in range(len(tracks)):
        lat = wgs84.subpoint_of(tracks[track]).latitude
        lon = wgs84.subpoint_of(tracks[track]).longitude
        relative_measure = (starlinks[track] - pointofinterest).at(timebase)
        alt, az, distance1 = relative_measure.altaz()
        altitude = alt.degrees
        azimuth = az.degrees
        relative_range = distance1.km
        plot_points = []
        plot_times = []
        az_trim = []
        alt_trim = []
        range_trim = []
        for point in range(len(lat.degrees)):
            if altitude[point] >= 0.0:
                distance_trim[track, point] = relative_range[point]

    def get_indices_of_k_smallest(arr, k):
        idx = np.argpartition(arr.ravel(), k)
        return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])


    indexholder = np.zeros((num_satellites, len(timebase)), dtype=int)
    for point in range(len(timebase)):
        indexholder[:, point] = get_indices_of_k_smallest(distance_trim[:, point], num_satellites)[0]

    return starlinks, indexholder

starlinks, indexholder = starlink_tracks(BristolPhysicsObservatory, t, num_satellites=20)

tracks = []
for sat in starlinks:
    pos = sat.at(t)
    tracks.append(pos)
    latcheck = np.any(np.isnan(wgs84.subpoint_of(pos).latitude.degrees))
    loncheck = np.any(np.isnan(wgs84.subpoint_of(pos).longitude.degrees))
    if latcheck | loncheck:
        tracks.pop(-1)
        print("caught one")

for track in range(len(tracks)):
    lat = wgs84.subpoint_of(tracks[track]).latitude
    lon = wgs84.subpoint_of(tracks[track]).longitude
    relative_measure = (starlinks[track] - BristolPhysicsObservatory).at(t)
    alt, az, distance = relative_measure.altaz()
    altitude = alt.degrees
    azimuth = az.degrees
    relative_range = distance.km
    plot_points = []
    plot_times = []
    az_trim = []
    alt_trim = []
    range_trim = []
    for point in range(len(lat.degrees)):
        if altitude[point] >= 0.0 and np.any(np.isin(track, indexholder[:, point])):
            plot_points.append((lon.degrees[point], lat.degrees[point]))
            plot_times.append(timebase[point].strftime("%Y-%m-%d %H:%M:%S"))
            az_trim.append(azimuth[point])
            alt_trim.append(altitude[point])
            range_trim.append(relative_range[point])

    for time_index, time in enumerate(plot_times):
        geo_features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": starlinks[track].name,
                    "style": {"color": colour[track], "weight": 6},
                    "times": [time] * len(plot_points[: time_index + 1][-trail_Size:]),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": plot_points[: time_index + 1][-trail_Size:],
                },
            }
        )
        geo_features.append(
            {
                "type": "Feature",
                "properties": {
                    "icon": "marker",
                    "iconstyle": {
                        "iconUrl": f"http://icons.iconarchive.com/icons/google/noto-emoji-travel-places/1024/42597-satellite-icon.png",
                        "iconSize": [24, 24],
                        "fillOpacity": 1,
                        "popupAnchor": [1, -17],
                    },
                    "popup": "Satellite: " + starlinks[track].name + "<dd>"
                                                                     "Latitude: "
                             + "%s" % float("%.8g" % plot_points[time_index][0])
                             + "<dd>"
                               "Longitude: "
                             + "%s" % float("%.8g" % plot_points[time_index][1])
                             + "<dd>"  # rounding 8 sigfigs
                               "Azimuth: "
                             + "%s" % float("%.8g" % az_trim[time_index])
                             + "°"
                             + "<dd>"
                               "Altitude: "
                             + "%s" % float("%.8g" % alt_trim[time_index])
                             + "°"
                             + "<dd>"
                               "Slant Range: "
                             + "%s" % float("%.8g" % range_trim[time_index])
                             + "km",
                    "name": starlinks[track].name,
                    "style": {"color": "black", "weight": 2},
                    "times": [time],
                },
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [plot_points[time_index]],
                },
            }
        )

# plot receiver
r_track = []
for time in range(len(t)):
    pos = BristolPhysicsObservatory.at(t)
    r_track.append(pos)

for track in range(len(r_track)):
    lat = wgs84.subpoint_of(r_track[track]).latitude
    lon = wgs84.subpoint_of(r_track[track]).longitude
    relative_measure = (receiver - BristolPhysicsObservatory).at(t)
    alt, az, distance = relative_measure.altaz()
    altitude = alt.degrees
    azimuth = az.degrees
    relative_range = distance.km
    plot_points = []
    plot_times = []
    az_trim = []
    alt_trim = []
    range_trim = []
    for point in range(len(lat.degrees)):
        # if altitude[point] >= 0.0:
        plot_points.append((lon.degrees[point], lat.degrees[point]))
        plot_times.append(timebase[point].strftime("%Y-%m-%d %H:%M:%S"))
        az_trim.append(azimuth[point])
        alt_trim.append(altitude[point])
        range_trim.append(relative_range[point])

    for time_index, time in enumerate(plot_times):
        geo_features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": "Ground Station at Bristol Physics Observatory",
                    "style": {"color": colour[track], "weight": 6},
                    "times": [time] * len(plot_points[: time_index + 1][-trail_Size:]),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": plot_points[: time_index + 1][-trail_Size:],
                },
            }
        )
        geo_features.append(
            {
                "type": "Feature",
                "properties": {
                    "icon": "marker",
                    "iconstyle": {
                        "iconUrl": f"https://www.bristol.ac.uk/media-library/sites/physics/new-images/501_10013623.jpg",
                        "iconSize": [64, 32],
                        "fillOpacity": 1,
                        "popupAnchor": [1, -17],
                    },
                    "popup": "Ground Station at Bristol Physics Observatory" + "<dd>"
                                                                               "Latitude: "
                             + "%s" % float("%.8g" % plot_points[time_index][0])
                             + "<dd>"
                               "Longitude: "
                             + "%s" % float("%.8g" % plot_points[time_index][1])
                             + "<dd>"  # rounding 8 sigfigs
                               "Azimuth: "
                             + "%s" % float("%.8g" % az_trim[time_index])
                             + "°"
                             + "<dd>"
                               "Altitude: "
                             + "%s" % float("%.8g" % alt_trim[time_index])
                             + "°"
                             + "<dd>"
                               "Slant Range: "
                             + "%s" % float("%.8g" % range_trim[time_index])
                             + "km",
                    "name": "Ground Station at Bristol Physics Observatory",
                    "style": {"color": "black", "weight": 2},
                    "times": [time],
                },
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [plot_points[time_index]],
                },
            }
        )

from folium.plugins import TimestampedGeoJson, Fullscreen

Fullscreen().add_to(m)

import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, MultiPolygon


def collec_to_gdf(collec_poly):
    """Transform a `matplotlib.contour.QuadContourSet` to a GeoDataFrame"""
    polygons, colors = [], []
    for i, polygon in enumerate(collec_poly.collections):
        mpoly = []
        for path in polygon.get_paths():
            try:
                path.should_simplify = False
                poly = path.to_polygons()
                # Each polygon should contain an exterior ring + maybe hole(s):
                exterior, holes = [], []
                if len(poly) > 0 and len(poly[0]) > 3:
                    # The first of the list is the exterior ring :
                    exterior = poly[0]
                    # Other(s) are hole(s):
                    if len(poly) > 1:
                        holes = [h for h in poly[1:] if len(h) > 3]
                mpoly.append(Polygon(exterior, holes))
            except:
                print('Warning: Geometry error when making polygon #{}'
                      .format(i))
        if len(mpoly) > 1:
            mpoly = MultiPolygon(mpoly)
            polygons.append(mpoly)
            colors.append(polygon.get_facecolor().tolist()[0])
        elif len(mpoly) == 1:
            polygons.append(mpoly[0])
            colors.append(polygon.get_facecolor().tolist()[0])
    return GeoDataFrame(
        geometry=polygons,
        data={'RGBA': colors},
        crs={'init': 'epsg:4326'})

# %%
# The Results
# ~~~~~~~~~~~~~~

TimestampedGeoJson(
    data=geo_json,
    transition_time=200,
    auto_play=True,
    add_last_point=False,
    period="PT1S",
    duration="PT0S",
).add_to(m)

# %%


m.show_in_browser()





