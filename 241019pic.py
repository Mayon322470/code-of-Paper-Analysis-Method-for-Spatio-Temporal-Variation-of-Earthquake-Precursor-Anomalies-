# !/D/Python3
# -*- coding:utf-8 -*-
# author: Yongsheng Ma
# time: 2023/8/23
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import csv

# Load the data from the CSV file
latitudes = []
longitudes = []
magnitudes = []

with open('earthquake_data.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip the header
    for row in csvreader:
        latitudes.append(float(row[0]))
        longitudes.append(float(row[1]))
        magnitudes.append(float(row[2]))

# Convert lists to numpy arrays
latitudes = np.array(latitudes)
longitudes = np.array(longitudes)
magnitudes = np.array(magnitudes)

# Create the map using Cartopy and add Stamen Terrain tiles
stamen_terrain = cimgt.Stamen('terrain-background')
fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': stamen_terrain.crs})
ax.set_extent([-73, -68, -23, -18])  # Set the map extent
ax.add_image(stamen_terrain, 9)  # The number is the zoom level

# Add geographic features
ax.add_feature(cfeature.COASTLINE.with_scale('10m'))  # High resolution coastline
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')  # High resolution country borders
ax.add_feature(cfeature.LAKES.with_scale('10m'), edgecolor='black')  # High resolution lakes
ax.add_feature(cfeature.RIVERS.with_scale('10m'))  # High resolution rivers
ax.add_feature(cfeature.OCEAN.with_scale('10m'))  # High resolution oceans
#ax.add_feature(cfeature.POPULATED_PLACES.with_scale('10m'), markersize=2)  # High resolution populated places

# Use a colormap to color the earthquake locations based on magnitude
cmap = plt.cm.get_cmap("viridis")
norm = plt.Normalize(min(magnitudes), max(magnitudes))
colors = cmap(norm(magnitudes))
sc = ax.scatter(longitudes, latitudes, c=colors, s=20, alpha=0.7, edgecolors="w", linewidth=0.5, transform=ccrs.PlateCarree())
ax.scatter()  # 本行把台站文件读取的信息传进来画到同一个坐标系上，应该就可以交差了。

# 以上，是绘图的主要逻辑，下面全都是辅助表示的代码，不需要任何添加和更改
# Add a colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.5)
cbar.set_label('Magnitude')

# Add gridlines and title
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_title("Enhanced Earthquake Locations Map")

# Save and show the map
plt.savefig("enhanced_earthquake_map_cartopy_v2.png", dpi=300)
plt.show()


