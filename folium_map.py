# THIS WAS JUST AN EXPERIMENT WITH A FOLIUM MAP



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