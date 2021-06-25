import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

img = cv2.imread('IMG_4767.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res = cv2.resize(img, (224, 224))
# res = Image.fromarray(img)
# res = res.resize((224, 224), Image.BILINEAR)

title = ['Ori', 'resize']
image = [img, res]
count = 2

# arr = np.asarray(res)
# arr = res.reshape(res.shape[0], -1)
# print(arr)
# df = pd.DataFrame(arr)
# df.to_csv('pix.csv')
# width, height = res.size
# format = res.format
# mode = res.mode

for i in range(count):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    plt.imshow(image[i])

plt.show()
# hasil = 'resize.jpg'
print(res[107:112, 90:95])
# cv2.imwrite(hasil, res)
# img_file = Image.open("resize.jpg")

# # load the pixel info
# pix = img_file.load()

# # get a tuple of the x and y dimensions of the image
# width, height = img_file.size

# # open a file to write the pixel data
# with open('output.csv', 'w+') as f:
#     f.write('R,G,B\n')

#     # read the details of each pixel and write them to the file
#     for x in range(width):
#         for y in range(height):
#             r = pix[x, y][0]
#             g = pix[x, x][1]
#             b = pix[x, x][2]
#             f.write('{0},{1},{2}\n'.format(r, g, b))
