import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import HourLocator, DateFormatter

from skyfield.api import load, EarthSatellite, wgs84

from skyfield.api import load, wgs84

starlink_url = "https://celestrak.com/NORAD/elements/supplemental/starlink.txt"
starlinks = load.tle_file(starlink_url)

# Build the time range `t` over which to plot, plus other values.

ts = load.timescale()

# Load the satellite's TLE entry.

astra2e = EarthSatellite(
    "1 39285U 13056A   22137.15428389  .00000157  00000+0  00000+0 0  9998",
    "2 39285   0.0686 331.1550 0002286  66.0071 281.6942  1.00269260 31625",
    "ASTRA 2E",
    ts,
)
astra2f = EarthSatellite(
    "1 38778U 12051A   22142.82294109  .00000140  00000+0  00000+0 0  9991",
    "2 38778   0.0565 240.3451 0000453 298.7811  25.7511  1.00269234 26228",
    "ASTRA 2F",
    ts,
)

astra2g = EarthSatellite(
    "1 40364U 14089A   22142.53592181  .00000140  00000+0  00000+0 0  9997",
    "2 40364   0.0579  63.3703 0005136 355.6036  42.3018  1.00273360 27060",
    "ASTRA 2G",
    ts,
)

ISS = EarthSatellite(
    "1 25544U 98067A   22143.37738542  .00010371  00000+0  19191-3 0  9996",
    "2 25544  51.6438  99.4639 0004967 147.2337 318.3670 15.49625671341330",
    "ISS (ZARYA)",
    ts,
)
satellites = [astra2e, astra2f, astra2g, ISS]
numberofstarlinks = len(starlinks) # total of 2386
for sat in range(numberofstarlinks):
    satellites.append(starlinks[sat])
# Compute geocentric positions for the satellite.
RoyalObservatoryGreenwich = wgs84.latlon(+51.476852, -0.0005, 45)

t0 = ts.utc(2022, 1, 23)
t1 = ts.utc(2022, 1, 24)

time_points = 5 * 60   # track over 5 hours
# change over time
t = ts.utc(2022, 5, 23, 13, range(0, time_points))
timebase = t.utc_datetime()

# satellites=[satellite]
# do for all starlink satellites
tracks = []
for sat in satellites:
    pos = sat.at(t)
    tracks.append(pos)
    latcheck = np.any(np.isnan(wgs84.subpoint_of(pos).latitude.degrees))
    loncheck = np.any(np.isnan(wgs84.subpoint_of(pos).longitude.degrees))
    if latcheck | loncheck:
        tracks.pop(-1)
        print("caught one")

# map reference site to map satellite paths relative to the reference site
import folium

m = folium.Map(
    location=[
        RoyalObservatoryGreenwich.latitude.degrees,
        RoyalObservatoryGreenwich.longitude.degrees,
    ],
    zoom_start=2,
    tiles="Stamen Terrain",
)

folium.Marker(
    [
        RoyalObservatoryGreenwich.latitude.degrees,
        RoyalObservatoryGreenwich.longitude.degrees,
    ],
    tooltip="Royal Observatory Greenwich",
    icon=folium.Icon(
        icon="fa-circle", prefix="fa", color="red"  # Marker for Measurement Site
    ),
).add_to(m)

# %%
# GeoJSON
# ~~~~~~~
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
# Plotting Tracks
# ~~~~~~~~~~~~~~~
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

png = "./satelliteicon.png"
with open(png, "rb") as lf:
    # open in binary mode, read bytes, encode, decode obtained bytes as utf-8 string
    b64_content = base64.b64encode(lf.read()).decode("utf-8")
# we only care for sattelites above the horizon from the point of interest, in this example the Royal Observatory, so eliminate plot points with altitude of less than 0 degrees.
# I have no idea why, but for folium line points and markers, the normal order of latitude,longitude is reversed, unless I am missing a config change.
for track in range(len(tracks)):
    lat = wgs84.subpoint_of(tracks[track]).latitude
    lon = wgs84.subpoint_of(tracks[track]).longitude
    relative_measure = (satellites[track] - RoyalObservatoryGreenwich).at(t)
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
        if altitude[point] >= 0.0:
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
                    "name": satellites[track].name,
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
                    "popup": "Satellite: " + satellites[track].name + "<dd>"
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
                    "name": satellites[track].name,
                    "style": {"color": "black", "weight": 2},
                    "times": [time],
                },
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [plot_points[time_index]],
                },
            }
        )

# %%
# The Results
# ~~~~~~~~~~~

from folium.plugins import TimestampedGeoJson, Fullscreen

Fullscreen().add_to(m)

TimestampedGeoJson(
    data=geo_json,
    transition_time=200,
    auto_play=True,
    add_last_point=False,
    period="PT1M",
    duration="PT1S",
).add_to(m)

# %%

m.save("satellitetracks.html")
