
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
dfd = sys.path[0]
#base_dir = dfd+"\img"
base_dir = "./img"
#
path_test_image = os.path.join(base_dir, "wenzi2"
                                         ".jpg")
print(path_test_image)
#   定义图片路径
image_color = cv2.imread(path_test_image)
#  表示pic
new_shape = (image_color.shape[1]*2, image_color.shape[0]*2)
#  颠倒原图的宽高
image_color = cv2.resize(image_color, new_shape)
# 重新绘制原图的大小
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
# 获取 pic的色彩空间（gary）


###-----------------------    图片二值化   -------------------------------------------
'''
adapyiveThreshold自适应二值化函数 
第一个原始图像
第二个像素值上限
第三个自适应方法Adaptive Method: 
— cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值 
—cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权 重为一个高斯窗口
第四个值的赋值方法：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
第五个Block size:规定领域大小（一个正方形的领域）
第六个常数C，阈值等于均值或者加权值减去这个常数（为0相当于阈值 就是求得领域内均值或者加权值） 
这种方法理论上得到的效果更好，相当于在动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值。
cv2.THRESH_BINARY（黑白二值） 
? cv2.THRESH_BINARY_INV（黑白二值反转） 
? cv2.THRESH_TRUNC （得到的图像为多像素值） 
? cv2.THRESH_TOZERO 
? cv2.THRESH_TOZERO_INV 

'''
adaptive_threshold = cv2.adaptiveThreshold(
image,
255,
cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY_INV, 11, 2)
#  255 位像素上线，ADAPTIVE_THRESH_MEAN_C ：领域内均值 ，cv2.THRESH_BINARY_INV（黑白二值反转），
cv2.imshow('binary image', adaptive_threshold)
cv2.waitKey(0)
# opencv图像自适应阈值化
#data_mix = np.array(adaptive_threshold)
#print(data_mix)

###-----------------------    提取每行  -------------------------------------------
# 矩阵压缩
horizontal_sum = np.sum(adaptive_threshold, axis=1)
# sum 表示矩阵元素相加， axis为0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
# 当axis为1时,是压缩列,即将每一行的元素相加,将矩阵压缩为一列
# 该处是将值压缩为一列
plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))  # shape0 为图像高度  917
# plot 做一个线图
plt.gca().invert_yaxis()
# gca 返回axes（轴线） gcf 返回figure(图）
plt.show()
# 显示表格


###-----------------------   提取峰值   -------------------------------------------
def extract_peek_ranges_from_array(array_vals, minimun_val=50, minimun_range=10):
    # array_vals 为峰值列表 minimun 和 minimun_range用于去除噪音 minimun_val 限制峰值最低限度 即分开字体所需要的  minimun_range限制字体不能太小
    start_i = None
    end_i = None
    peek_ranges = []
    # enumerate 返回 索引号与索引值
    for i, val in enumerate(array_vals): # i 是索引号 ， val 是值
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges
#  peel_range 返回的是字体所在行的位置范围。

peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
# 调用获取行范围

###-----------------------   划分出每一行   -------------------------------------------
line_seg_adaptive_threshold = np.copy(adaptive_threshold)
# 赋值一个二值化图像
for i, peek_range in enumerate(peek_ranges):
    x = 0
    y = peek_range[0]
    # y表示一行顶部的位置
    w = line_seg_adaptive_threshold.shape[1]
    # 获得二值化图像的宽
    h = peek_range[1] - y
    # 计算行高
    pt1 = (x, y)
    # 一行左上角位置
    pt2 = (x + w, y + h)
    # 一行右下角位置
    cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
    # 画除矩阵区域  第一个参数：img  第二个：左上角位置 三：右下角位置 四：区域边界颜色  五：边界宽度
# 显示二值化图像
cv2.imshow('line image', line_seg_adaptive_threshold)
cv2.waitKey(0)



vertical_peek_ranges2d = []
for peek_range in peek_ranges:
    start_y = peek_range[0]
    end_y = peek_range[1]
    line_img = adaptive_threshold[start_y:end_y, :]
    vertical_sum = np.sum(line_img, axis=0)
    vertical_peek_ranges = extract_peek_ranges_from_array(
        vertical_sum,
        minimun_val=40,
        minimun_range=1)
    vertical_peek_ranges2d.append(vertical_peek_ranges)

## Draw
color = (0, 0, 255)
for i, peek_range in enumerate(peek_ranges):
    for vertical_range in vertical_peek_ranges2d[i]:
        x = vertical_range[0]
        y = peek_range[0]
        w = vertical_range[1] - x
        h = peek_range[1] - y
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(image_color, pt1, pt2, color)


        def median_split_ranges(peek_ranges):
            new_peek_ranges = []
            widthes = []
            for peek_range in peek_ranges:
                w = peek_range[1] - peek_range[0] + 1
                widthes.append(w)
            widthes = np.asarray(widthes)
            median_w = np.median(widthes)
            for i, peek_range in enumerate(peek_ranges):
                num_char = int(round(widthes[i] / median_w, 0))
                if num_char > 1:
                    char_w = float(widthes[i] / num_char)
                    for i in range(num_char):
                        start_point = peek_range[0] + int(i * char_w)
                        end_point = peek_range[0] + int((i + 1) * char_w)
                        new_peek_ranges.append((start_point, end_point))
                else:
                    new_peek_ranges.append(peek_range)
            return new_peek_ranges


        vertical_peek_ranges2d = []
        for peek_range in peek_ranges:
            start_y = peek_range[0]
            end_y = peek_range[1]
            line_img = adaptive_threshold[start_y:end_y, :]
            vertical_sum = np.sum(line_img, axis=0)
            vertical_peek_ranges = extract_peek_ranges_from_array(
                vertical_sum,
                minimun_val=40,
                minimun_range=1)
            vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
            vertical_peek_ranges2d.append(vertical_peek_ranges)

        ## Draw
        color = (0, 0, 255)
        for i, peek_range in enumerate(peek_ranges):
            for vertical_range in vertical_peek_ranges2d[i]:
                x = vertical_range[0]
                y = peek_range[0]
                w = vertical_range[1] - x
                h = peek_range[1] - y
                pt1 = (x, y)
                pt2 = (x + w, y + h)
                cv2.rectangle(image_color, pt1, pt2, color)
        cv2.imshow('', image_color)
        cv2.waitKey(1)

cv2.destroyAllWindows()
cv2.imshow('char image', image_color)
cv2.waitKey(0)




