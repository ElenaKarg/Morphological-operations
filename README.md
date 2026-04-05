# Morphological-operations
Dilation, Erosion, Opening and Closing

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- Load color image ----
image = cv2.imread('C:/Users/elena/Desktop/bubbles.jpg')
if image is None:
    raise ValueError("Image not found. Check path!")

# Convert for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- Kernel ----
kernel = np.ones((3, 3), np.uint8)

# ---- Morphological operations ----
dilated = cv2.dilate(image, kernel, iterations=5)
eroded = cv2.erode(image, kernel, iterations=5)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# ---- Convert results to RGB for matplotlib ----
dilated_rgb = cv2.cvtColor(dilated, cv2.COLOR_BGR2RGB)
eroded_rgb = cv2.cvtColor(eroded, cv2.COLOR_BGR2RGB)
opening_rgb = cv2.cvtColor(opening, cv2.COLOR_BGR2RGB)
closing_rgb = cv2.cvtColor(closing, cv2.COLOR_BGR2RGB)

# ---- Visualization ----
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(image_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(232)
plt.imshow(dilated_rgb)
plt.title('Dilated')
plt.axis('off')

plt.subplot(233)
plt.imshow(eroded_rgb)
plt.title('Eroded')
plt.axis('off')

plt.subplot(234)
plt.imshow(opening_rgb)
plt.title('Opening')
plt.axis('off')

plt.subplot(235)
plt.imshow(closing_rgb)
plt.title('Closing')
plt.axis('off')

plt.tight_layout()
plt.show()
