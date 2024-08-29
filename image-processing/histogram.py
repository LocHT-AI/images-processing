import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("xam.jpg", 0)
img_2=cv2.imread("gray.jpeg",0)


def compute_hist(img):
    hist = np.zeros((256,), np.uint8)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] += 1
    return hist

def equal_hist(hist):
    cumulator = np.zeros_like(hist, np.float64)
    for i in range(len(cumulator)):
        cumulator[i] = hist[:i].sum()
    print(cumulator)
    new_hist = (cumulator - cumulator.min())/(cumulator.max() - cumulator.min()) * 255
    new_hist = np.uint8(new_hist)
    return new_hist

hist = compute_hist(img).ravel()
new_hist = equal_hist(hist)

h, w = img.shape[:2]
for i in range(h):
   for j in range(w):
       img[i,j] = new_hist[img[i,j]]
       
fig = plt.figure()
ax = plt.subplot(121)
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.plot(hist)
plt.plot(new_hist)
plt.show()
