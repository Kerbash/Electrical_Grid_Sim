"""
Using the built in python library pyTorch, two different simulation functions are performed.
One with and without the use of PinnedMemory. The pre-calculated electrical usage are loaded from pickle
files. From there the function is then called to pool the matrix into a smaller matrix for rendering.

Two different version of the pyTorch function is used. The first is the Naive version in which the
function max_pool2d is used. The second is, also used the max_pool2d function, but with an added pinned
memory before initializing the GPU. The pinned memory is used to transfer the data from the CPU to the
GPU. This is done to avoid the overhead of the CPU to GPU transfer.

The results are then retrived. Using the .cpu() function to transfer the data from the GPU to the CPU.
"""

# THIS WAS JUST AN EXPERIMENT WITH A FOLIUM MAP
# uses GeoPandas too
# Problem: we would need shapefiles to make this work for a specific location
# We would also need to have longitude and latitude coordinates for the houses to plot them.

# +++++++++++++++++ INSTALLATION INSTRUCTIONS +++++++++++++++++
# https://geopandas.org/en/stable/getting_started/install.html
# https://forrest.nyc/get-started-with-python-and-geopandas-in-3-minutes/   <<< pip install
# conda create -n geo_env
# conda activate geo_env
# conda config --env --add channels conda-forge
# conda config --env --set channel_priority strict
# conda install python=3 geopandas
# $ conda install geopandas
# $ conda install pandas fiona shapely pyproj rtree


import geopandas

# https://geopandas.org/en/stable/docs/user_guide/interactive_mapping.html
nybb = geopandas.read_file(geopandas.datasets.get_path('nybb'))
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
cities = geopandas.read_file(geopandas.datasets.get_path('naturalearth_cities'))

# https://python-visualization.github.io/folium/
# $ conda install folium
import folium
import matplotlib.pyplot as plt
from mapclassify import classify

nybb.explore(
    column="BoroName",  # make choropleth based on "BoroName" column
    tooltip="BoroName",  # show "BoroName" value in tooltip (on hover)
    popup=True,  # show all values in popup (on click)
    tiles="CartoDB positron",  # use "CartoDB positron" tiles
    cmap="Set1",  # use "Set1" matplotlib colormap
    style_kwds=dict(color="black"),  # use black outline
)

#explore() returns folium.Map object
m = world.explore(
     column="pop_est",  # make choropleth based on "BoroName" column
     scheme="naturalbreaks",  # use mapclassify's natural breaks scheme
     legend=True, # show legend
     k=10, # use 10 bins
     legend_kwds=dict(colorbar=False), # do not use colorbar
     name="countries" # name of the layer in the map
)

cities.explore(
     m=m, # pass the map object
     color="red", # use red color on all points
     marker_kwds=dict(radius=10, fill=True), # make marker radius 10px with fill
     tooltip="name", # show "name" column in the tooltip
     tooltip_kwds=dict(labels=False), # do not show column label in the tooltip
     name="cities" # name of the layer in the map
)

folium.TileLayer('Stamen Toner', control=True).add_to(m)  # use folium to add alternative tiles
folium.LayerControl().add_to(m)  # use folium to add layer control

m  # show map