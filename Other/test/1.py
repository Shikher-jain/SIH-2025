import rasterio
import numpy as np

# Path to your .tif file

# tif_path = "agro final/data/cell_1_2/test-1_sensor_2025-09-23T13-56-06-245Z.tif"
# tif_path ="C:\shikher_jain\SIH\merged_analysis.tif"
# tif_path="C:\shikher_jain\SIH\HectFarm-1000m_satellite_2025-09-25_2025-09-27T15-07-08-280Z.tif"
# tif_path="C:\shikher_jain\SIH\HectFarm-1000m_sensor_2025-09-27T15-07-18-672Z.tif"
# tif_path="C:\shikher_jain\SIH\merged_analysis.tif"

tif_path = r"C:\shikher_jain\SIH\test1\UttarPradesh_Test_Tiny_stacked_26bands_2025-09-27T22-54-02-251Z.tif"

with rasterio.open(tif_path) as src:
    # for i in range(1,27):
    #     arr = src.read(i)  # Read first band as numpy array

    #     print(f"Numpy array shape:{i}", arr.shape)
    #     print(arr[:12][:12])
    
    i=24
    arr = src.read(i)    # Read first band as numpy array
    np.set_printoptions(threshold=np.inf)

    print(f"Numpy array shape:{i}", arr.shape)  #512 x 512
    print(arr[:12][:12])
