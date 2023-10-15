####################
# 创建图形
####################

import numpy as np
import matplotlib.pyplot as plt

# 创建一个黑色背景的图片
img_size = 512
img = np.zeros( (img_size, img_size) )

# 计算正方形的左上角和右下角坐标
square_size = 16
square_top_left = (img_size - square_size) // 2
square_bottom_right = square_top_left + square_size

# 在图片中央绘制白色正方形
img[ square_top_left:square_bottom_right,
     square_top_left:square_bottom_right ] = 1

####################
# 三维可视化
####################

from scipy import ndimage
from matplotlib.widgets import Slider

# 显示原始图片和傅里叶变换后的振幅谱
fig, ax = plt.subplots( 1, 2, figsize=(12, 6) )
plt.subplots_adjust( bottom=0.2 )  # 给下面留出一块

ax_layer = plt.axes( [0.1, 0.1, 0.8, 0.03] )
slider_layer = Slider( ax_layer, 'layer',
                       0, 180,
                       valfmt='%d',
                       valinit=180//2, valstep=1,
                       )



def update(angle):

    # order 是不同的插值方法，0是最近邻，1是双线性，3是双三次
    # 0会导致出现尖锐边缘，傅里叶变换之后的结果更艺术一些
    rotated_img = ndimage.rotate( img, angle,
                                  reshape=False,
                                  order=0,  # 产生尖锐边缘，傅里叶变换频谱泄露
                                  mode='constant', cval=0,
                                  )

    ####################
    # 傅里叶变换
    ####################

    # 进行傅里叶变换
    img_fft = np.fft.fft2(rotated_img)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # 计算振幅谱
    magnitude_spectrum = np.abs(img_fft_shifted)

    plt.subplot(121)
    ax[0].clear()
    plt.imshow(rotated_img, cmap='gray')
    plt.title('Original img')
    plt.axis('off')

    plt.subplot(122)
    ax[1].clear()
    plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray')
    plt.title('Magnitude Spectrum (log scale)')
    plt.axis('off')
    
    fig.canvas.draw_idle()

slider_layer.on_changed(update)
slider_layer.reset()
slider_layer.set_val( 180//2 )

plt.show()
