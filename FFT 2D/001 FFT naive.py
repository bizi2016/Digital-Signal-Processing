####################
# 创建图形
####################

import numpy as np
import matplotlib.pyplot as plt

# 创建一个黑色背景的图片
img_size = 1024
img = np.zeros( (img_size, img_size) )

# 计算正方形的左上角和右下角坐标
square_size = 64
square_top_left = (img_size - square_size) // 2
square_bottom_right = square_top_left + square_size

# 在图片中央绘制白色正方形
img[ square_top_left:square_bottom_right,
     square_top_left:square_bottom_right ] = 1

####################
# 傅里叶变换
####################

# 进行傅里叶变换
img_fft = np.fft.fft2(img)
img_fft_shifted = np.fft.fftshift(img_fft)

# 计算振幅谱
magnitude_spectrum = np.abs(img_fft_shifted)

# 显示原始图片和傅里叶变换后的振幅谱
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original img')
plt.axis('off')

plt.subplot(122)
plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
plt.title('Magnitude Spectrum (log scale)')
plt.axis('off')

plt.tight_layout()
plt.show()
