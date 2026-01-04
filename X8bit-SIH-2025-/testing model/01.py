from scipy.io import loadmat

data = loadmat('Indian_pines_corrected.mat')
image = data['indian_pines_corrected']  # shape: (145, 145, 220)

print("Image shape:", image.shape)


# =================================================================

from spectral import *

img = open_image('Indian_pines_corrected.hdr')  # paired with .dat
data = img.load()  # shape: (145, 145, 220)

# View one band
band50 = data[:,:,50]
imshow(band50, title="Band 50")

# View RGB composite
view = imshow(data, bands=[29, 19, 9])
