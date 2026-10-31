import numpy as np

latitudes = np.arange(7.5, 38.0, 1.0)
longitudes = np.arange(67.5, 98.0, 1.0)

lat_mask = (latitudes >= 15.5) & (latitudes <= 22.1)
lon_mask = (longitudes >= 72.6) & (longitudes <= 80.9)

print("Maharashtra latitude points:", latitudes[lat_mask])
print("Maharashtra longitude points:", longitudes[lon_mask])
print("Grid shape inside bounding box:",
      (np.count_nonzero(lat_mask), np.count_nonzero(lon_mask)))
print("Total grid points:", np.count_nonzero(lat_mask) *
      np.count_nonzero(lon_mask))