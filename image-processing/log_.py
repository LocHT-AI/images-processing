import cv2
import numpy as np

img = cv2.imread('2.jpg')
img=img[::2,::2]
c = 255 / np.log(1 + np.max(img))
log_transformed = c * np.log(1 + img)
log_transformed = np.array(log_transformed, dtype=np.uint8)
# cv2.imwrite('log_transformed_image.jpg', log_transformed)
cv2.imshow("ori",img)
cv2.imshow("log_transformed",log_transformed)
# Wait for a key event and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()