import os

import numpy as np
import math

import pandas as pd
import torch
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

conv_ops_count = 0
p_compare_count = 0
# BN 用到的参数
epsilon = 0.00001
# BN 参数的最大范围，目前留有很大富裕，模型越大，这个也越大
MAX_SHORT = 32767
MIN_SHORT = -32768

max = 0
min = 9299


def np2txt(tensor, name):
    # 检查张量的维度
    dims = tensor.ndim

    # 根据张量的维度进行不同的处理
    if dims == 2:
        # 二维张量直接保存
        np.savetxt(f"./middle/{name}", tensor, fmt='%d')
        #print(f"Saved ./middle/{name} with shape {tensor.shape}")
    elif dims == 3:
        # 三维张量展开成多个二维张量并保存在一个文件中
        with open(f"./middle/{name}", "w") as f:
            for i in range(tensor.shape[2]):
                f.write(f"Slice {i} with shape {tensor[:, :, i].shape}\n")
                np.savetxt(f, tensor[:, :, i], fmt='%d')
                f.write("\n")  # 添加空行以分隔不同的切片
        #print(f"Saved ./middle/{name} with {tensor.shape[2]} slices")
    elif dims == 4:
        # 四维张量先展开成多个三维张量，然后每个三维张量再展开成多个二维张量，保存在一个文件中
        with open(f"./middle/{name}", "w") as f:
            for i in range(tensor.shape[3]):
                for j in range(tensor.shape[0]):
                    f.write(f"Slice {i}_{j} with shape {tensor[j, :, :, i].shape}\n")
                    np.savetxt(f, tensor[j, :, :, i], fmt='%d')
                    f.write("\n")  # 添加空行以分隔不同的切片
        #print(f"Saved ./middle/{name} with {tensor.shape[3]} groups and {tensor.shape[0]} slices per group")
    else:
        print("Unsupported tensor dimension")


def img2col_2d(input_array, kernel_size):
    # Extracting the dimensions of the input array
    input_height, input_width = input_array.shape

    # Output dimensions
    output_height = input_height - kernel_size + 1
    output_width = input_width - kernel_size + 1

    # Initialize the output array
    col_array = np.zeros((output_height * output_width, kernel_size * kernel_size))

    col_row = 0
    for y in range(0, output_height):
        for x in range(0, output_width):
            window = input_array[y:y + kernel_size, x:x + kernel_size].flatten()
            col_array[col_row, :] = window
            col_row += 1

    return col_array


def img2col_1d(input_array, kernel_size):
    input_len = input_array.shape

    # Output dimensions
    output_len = input_len[0] - kernel_size + 1

    # 构建输出结果
    output_list = []
    for i in range(0, output_len):
        output_item = input_array[i:(i + kernel_size)]
        output_list.append(output_item)

    output_result = np.stack(output_list)
    return output_result


def mulMatrix(m1, m2):
    # 在这里统计以下矩阵乘法的运算次数
    # global conv_ops_count
    # M, K = m1.shape
    # N = m2.shape[1]
    # flops = M * N
    # conv_ops_count += flops
    return np.dot(m1, m2)


def kernel_reshape_2d(matrices):
    # 确保输入是一个三维数组，并且第二和第三维度相等
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("Input must be a 3D array with the second and third dimensions equal")
    # 获取 n 的值（第二和第三维度的大小）
    m, n, _ = matrices.shape
    # 按行展开每个 n*n 矩阵，得到 m 个 n^2*1 的列向量
    flattened_matrices = np.reshape(matrices, (m, n * n))
    # 转置每个列向量，使其成为 1*n^2 的行向量
    transposed_matrices = np.transpose(flattened_matrices)
    # 将 m 个行向量合并成一个 (n^2)*m 的矩阵
    combined_matrix = np.reshape(transposed_matrices, (n * n, m))
    return combined_matrix


def kernel_reshape_1d(matrices):
    # 确保输入是一个三维数组，并且第二和第三维度相等
    if matrices.ndim != 2:
        raise ValueError("Input must be a 2D array")
    # 获取 n 的值（第二和第三维度的大小）
    m, n = matrices.shape
    # 转置每个列向量，得到行向量
    transposed_matrices = np.transpose(matrices)
    # 将 m 个行向量合并成一个 n*m 的矩阵
    combined_matrix = np.reshape(transposed_matrices, (n, m))
    return combined_matrix


def conv_2d(iact, weight):
    # 获取输入和权重的维度
    m, n, n = iact.shape
    g, m, k, k = weight.shape

    # 输出特征图的尺寸
    output_height = n - k + 1
    output_width = n - k + 1

    # 初始化输出特征图
    output = np.zeros((output_height * output_width, g))

    # 对每个通道进行卷积 25 * n
    for ch in range(m):
        iact_sub = iact[ch, :, :]
        iact_sub_matrix = img2col_2d(iact_sub, 5)
        kernels = weight[:, ch, :, :]
        kernels_matrix = kernel_reshape_2d(kernels)
        # print(iact_sub_matrix.shape, " --- ", kernels_matrix.shape)
        output = output + mulMatrix(iact_sub_matrix, kernels_matrix)
    return output.reshape((output_height, output_width, g))


def conv_1d(iact, weight):
    # 获取输入和权重的维度
    m, n = iact.shape
    g, m, k = weight.shape

    # 输出特征图的尺寸
    output_len = n - k + 1

    # 初始化输出特征图
    output = np.zeros((output_len, g))

    # 对每个通道进行卷积 25 * n
    for ch in range(m):
        iact_sub = iact[ch, :]
        iact_sub_matrix = img2col_1d(iact_sub, 16)
        kernels = weight[:, ch, :]
        kernels_matrix = kernel_reshape_1d(kernels)
        # print(iact_sub_matrix.shape, " --- ", kernels_matrix.shape)
        output = output + mulMatrix(iact_sub_matrix, kernels_matrix)
    return output.reshape((output_len, g))


def max_pool2d(input, kernel_size, stride):
    in_height, in_width = input.shape
    out_height = (in_height - kernel_size) // stride + 1
    out_width = (in_width - kernel_size) // stride + 1

    output = np.zeros((out_height, out_width))

    for y in range(out_height):
        for x in range(out_width):
            window = input[y * stride:y * stride + kernel_size, x * stride:x * stride + kernel_size]
            output[y, x] = np.max(window)

    return output


def max_pool1d(input, kernel_size, stride):
    in_length = input.shape[0]  # 一维数组的长度
    out_length = (in_length - kernel_size) // stride + 1  # 输出数组的长度

    output = np.zeros(out_length)  # 初始化输出数组

    for i in range(out_length):
        # 计算当前窗口的起始和结束索引
        start_idx = i * stride
        end_idx = start_idx + kernel_size
        # 提取当前窗口的数据
        window = input[start_idx:end_idx]
        # 计算窗口中的最大值，并赋值给输出数组的对应位置
        output[i] = np.max(window)

    return output


def block_cal_2d(iact, block_conv_weights, block_bn_weights, block_bn_bias, block_bn_mean, block_bn_var, block_bn_relu,
                 pool_stride, level):
    global min
    global max
    # 先进行卷积和池化运算
    conv_result = conv_2d(iact, block_conv_weights)
    pool_result = list()
    w, h, chs = conv_result.shape
    for ch in range(chs):
        conv_item = conv_result[:, :, ch]
        pool_item = max_pool2d(conv_item, 2, pool_stride)
        pool_result.append(pool_item)

    ch_pool_result = np.stack(pool_result)

    # 获取最大值
    max_value = np.max(ch_pool_result)

    # 获取最小值
    min_value = np.min(ch_pool_result)

    # 获取值的分布情况
    unique_values, counts = np.unique(ch_pool_result, return_counts=True)

    if max_value > max:
        max = max_value

    if min_value < min:
        min = min_value


    # print("Pool: ", ch_pool_result.shape)
    ch_result = []
    BNP = getBNParam(block_bn_weights, block_bn_bias, block_bn_mean, block_bn_var, block_bn_relu)

    BNP_p1 = BNP[:, 0]
    BNP_p2 = BNP[:, 1]
    # 创建一个包含两个子图的图形，子图排列为1行2列
    # fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    # # 在第一个子图上绘制 np1 的箱状图，并设置箱体颜色为蓝色
    # ax[0].boxplot(ch_pool_result.ravel(), patch_artist=True, boxprops=dict(facecolor='blue'))
    # ax[0].set_title('Pool Out Data')
    # # 在第二个子图上绘制 np2 的箱状图，并设置箱体颜色为绿色
    # ax[1].boxplot(BNP_p1.ravel(), patch_artist=True, boxprops=dict(facecolor='green'))
    # ax[1].set_title('BatchNormal Params P1')
    # # 在第二个子图上绘制 np2 的箱状图，并设置箱体颜色为绿色
    # ax[2].boxplot(BNP_p2.ravel(), patch_artist=True, boxprops=dict(facecolor='red'))
    # ax[2].set_title('BatchNormal Params P2')
    # # 显示图形
    # plt.show()
    # 计算得到参数
    chs = block_bn_weights.shape
    for i in range(chs[0]):
        # a = block_bn_relu[0]
        # k = block_bn_weights[i] / math.sqrt(block_bn_var[i] + epsilon)
        # b = block_bn_bias[i] - ((block_bn_mean[i] * block_bn_weights[i]) / math.sqrt(block_bn_var[i] + epsilon))
        # p1 = -1 * (b / k)
        # p2 = -1 * (b / (a * k))
        p1 = BNP[i, 0]
        p2 = BNP[i, 1]
        af = BNP[i, 2]
        akf = BNP[i, 3]
        ch_y = ch_pool_result[i, :].copy()  # 创建ch_y的副本

        for m in range(ch_y.shape[0]):
            for n in range(ch_y.shape[1]):
                now = ch_y[m, n]
                if abs(now) > max:
                    max = abs(now)
                if now >= 0:
                    if af:
                        if now >= p1:
                            ch_y[m, n] = 1
                        else:
                            ch_y[m, n] = -1
                    else:
                        if now > p1:
                            ch_y[m, n] = -1
                        else:
                            ch_y[m, n] = 1
                else:
                    if akf > 0:
                        if now >= p2:
                            ch_y[m, n] = 1
                        else:
                            ch_y[m, n] = -1
                    else:
                        if now > p2:
                            ch_y[m, n] = -1
                        else:
                            ch_y[m, n] = 1

        ch_result.append(ch_y)

    ch_result = np.stack(ch_result)
    return ch_result


def block_cal_1d(iact, block_conv_weights, block_bn_weights, block_bn_bias, block_bn_mean, block_bn_var, block_bn_relu,
                 pool_stride, final):
    # 先进行卷积和池化运算
    global min
    global max
    conv_result = conv_1d(iact, block_conv_weights)
    pool_result = list()
    len, chs = conv_result.shape
    for ch in range(chs):
        conv_item = conv_result[:, ch]
        pool_item = max_pool1d(conv_item, 4, pool_stride)
        pool_result.append(pool_item)

    ch_pool_result = np.stack(pool_result)

    # 获取最大值
    max_value = np.max(ch_pool_result)

    # 获取最小值
    min_value = np.min(ch_pool_result)

    # # 直方图
    # import seaborn as sns
    # sns.kdeplot(ch_pool_result.ravel(), shade=True)
    # plt.title('Density Plot')
    # plt.show()

    if max_value > max:
        max = max_value

    if min_value < min:
        min = min_value


    if final:
        return ch_pool_result

    ch_result = []
    BNP = getBNParam(block_bn_weights, block_bn_bias, block_bn_mean, block_bn_var, block_bn_relu)

    BNP_p1 = BNP[:, 0]
    BNP_p2 = BNP[:, 1]
    # 创建一个包含两个子图的图形，子图排列为1行2列
    # fig, ax = plt.subplots(1, 3, figsize=(9, 5))
    # # 在第一个子图上绘制 np1 的箱状图，并设置箱体颜色为蓝色
    # ax[0].boxplot(ch_pool_result.ravel(), patch_artist=True, boxprops=dict(facecolor='blue'))
    # ax[0].set_title('Pool Out Data')
    # # 在第二个子图上绘制 np2 的箱状图，并设置箱体颜色为绿色
    # ax[1].boxplot(BNP_p1.ravel(), patch_artist=True, boxprops=dict(facecolor='green'))
    # ax[1].set_title('BatchNormal Params P1')
    # # 在第二个子图上绘制 np2 的箱状图，并设置箱体颜色为绿色
    # ax[2].boxplot(BNP_p2.ravel(), patch_artist=True, boxprops=dict(facecolor='red'))
    # ax[2].set_title('BatchNormal Params P2')
    # # 显示图形
    # plt.show()
    # 计算得到参数
    chs = block_bn_weights.shape
    for i in range(chs[0]):
        # a = block_bn_relu[0]
        # k = block_bn_weights[i] / math.sqrt(block_bn_var[i] + epsilon)
        # b = block_bn_bias[i] - ((block_bn_mean[i] * block_bn_weights[i]) / math.sqrt(block_bn_var[i] + epsilon))
        # p1 = -1 * (b / k)
        # p2 = -1 * (b / (a * k))
        p1 = BNP[i, 0]
        p2 = BNP[i, 1]
        af = BNP[i, 2]
        akf = BNP[i, 3]
        ch_y = ch_pool_result[i, :].copy()  # 创建ch_y的副本

        for m in range(ch_y.shape[0]):
            now = ch_y[m]
            if abs(now) > max:
                max = abs(now)
            if now >= 0:
                if af:
                    if now >= p1:
                        ch_y[m] = 1
                    else:
                        ch_y[m] = -1
                else:
                    if now > p1:
                        ch_y[m] = -1
                    else:
                        ch_y[m] = 1
            else:
                if akf:
                    if now >= p2:
                        ch_y[m] = 1
                    else:
                        ch_y[m] = -1
                else:
                    if now > p2:
                        ch_y[m] = -1
                    else:
                        ch_y[m] = 1

        ch_result.append(ch_y)

    ch_result = np.stack(ch_result)
    return ch_result


# 根据BN层训练出来的参数，计算阈值P1和P2
def getBNParam(block_bn_weights, block_bn_bias, block_bn_mean, block_bn_var, block_bn_relu):
    # 计算得到参数
    result = []
    chs = block_bn_weights.shape
    for i in range(chs[0]):
        a = block_bn_relu[0]
        k = block_bn_weights[i] / math.sqrt(block_bn_var[i] + epsilon)
        b = block_bn_bias[i] - ((block_bn_mean[i] * block_bn_weights[i]) / math.sqrt(block_bn_var[i] + epsilon))
        p1 = -1 * (b / k)
        p2 = -1 * (b / (a * k))

        # 算出来的P1和P2是很大的，这并不方便RTL实现，可以缩小点，因为计算产生的参数一般不会特别大
        if p1 > MAX_SHORT:
            p1 = MAX_SHORT
        elif p1 < MIN_SHORT:
            p1 = MIN_SHORT

        if p2 > MAX_SHORT:
            p2 = MAX_SHORT
        elif p2 < MIN_SHORT:
            p2 = MIN_SHORT

        af = k > 0
        akf = (a * k) > 0
        result.append([p1, p2, af, akf])

    result = np.stack(result)


    return result


# 根据2D卷积层训练出来的参数
def getConv2DParam(conv_weights):
    kernels_matrixs = []
    n, m, _, _ = conv_weights.shape
    for ch in range(m):
        kernels = conv_weights[:, ch, :, :]
        kernels_matrix = kernel_reshape_2d(kernels)
        kernels_matrixs.append(kernels_matrix)

    return np.stack(kernels_matrixs)


# 根据1D卷积层训练出来的参数
def getConv1DParam(conv_weights):
    kernels_matrixs = []
    n, m, _ = conv_weights.shape
    for ch in range(m):
        kernels = conv_weights[:, ch, :]
        kernels_matrix = kernel_reshape_1d(kernels)
        kernels_matrixs.append(kernels_matrix)

    return np.stack(kernels_matrixs)


def get_weights():
    count1 = 0
    count16 = 0
    block1_conv_weights_path = "./Weights/block1_conv_tensor.npy"
    block1_conv_weights = np.load(block1_conv_weights_path)
    block1_bn_weights_path = "./Weights/block1_bn_weight_tensor.npy"
    block1_bn_weights = np.load(block1_bn_weights_path)
    block1_bn_bias_path = "./Weights/block1_bn_bias_tensor.npy"
    block1_bn_bias = np.load(block1_bn_bias_path)
    block1_bn_mean_path = "./Weights/block1_bn_mean_tensor.npy"
    block1_bn_mean = np.load(block1_bn_mean_path)
    block1_bn_var_path = "./Weights/block1_bn_var_tensor.npy"
    block1_bn_var = np.load(block1_bn_var_path)
    block1_bn_relu_path = "./Weights/block1_prelu_tensor.npy"
    block1_bn_relu = np.load(block1_bn_relu_path)
    block1_BNParam = getBNParam(block1_bn_weights, block1_bn_bias, block1_bn_mean, block1_bn_var, block1_bn_relu)
    block1_ConvParam = getConv2DParam(block1_conv_weights)
    m, _ = block1_BNParam.shape
    count16 = count16 + (m*2)
    count1 = count1 + block1_ConvParam.size + (m*2)
    # count1 = count1 + block1_ConvParam.size
    np2txt(block1_BNParam, "block1_BNParam")
    np2txt(block1_ConvParam, "block1_ConvParam")

    block2_conv_weights_path = "./Weights/block2_conv_tensor.npy"
    block2_conv_weights = np.load(block2_conv_weights_path)
    block2_bn_weights_path = "./Weights/block2_bn_weight_tensor.npy"
    block2_bn_weights = np.load(block2_bn_weights_path)
    block2_bn_bias_path = "./Weights/block2_bn_bias_tensor.npy"
    block2_bn_bias = np.load(block2_bn_bias_path)
    block2_bn_mean_path = "./Weights/block2_bn_mean_tensor.npy"
    block2_bn_mean = np.load(block2_bn_mean_path)
    block2_bn_var_path = "./Weights/block2_bn_var_tensor.npy"
    block2_bn_var = np.load(block2_bn_var_path)
    block2_bn_relu_path = "./Weights/block2_prelu_tensor.npy"
    block2_bn_relu = np.load(block2_bn_relu_path)
    block2_BNParam = getBNParam(block2_bn_weights, block2_bn_bias, block2_bn_mean, block2_bn_var, block2_bn_relu)
    block2_ConvParam = getConv2DParam(block2_conv_weights)
    m, _ = block2_BNParam.shape
    count16 = count16 + (m * 2)
    count1 = count1 + block2_ConvParam.size + (m*2)
    # count1 = count1 + block2_ConvParam.size
    np2txt(block2_BNParam, "block2_BNParam")
    np2txt(block2_ConvParam, "block2_ConvParam")

    block3_conv_weights_path = "./Weights/block3_conv_tensor.npy"
    block3_conv_weights = np.load(block3_conv_weights_path)
    block3_bn_weights_path = "./Weights/block3_bn_weight_tensor.npy"
    block3_bn_weights = np.load(block3_bn_weights_path)
    block3_bn_bias_path = "./Weights/block3_bn_bias_tensor.npy"
    block3_bn_bias = np.load(block3_bn_bias_path)
    block3_bn_mean_path = "./Weights/block3_bn_mean_tensor.npy"
    block3_bn_mean = np.load(block3_bn_mean_path)
    block3_bn_var_path = "./Weights/block3_bn_var_tensor.npy"
    block3_bn_var = np.load(block3_bn_var_path)
    block3_bn_relu_path = "./Weights/block3_prelu_tensor.npy"
    block3_bn_relu = np.load(block3_bn_relu_path)
    block3_BNParam = getBNParam(block3_bn_weights, block3_bn_bias, block3_bn_mean, block3_bn_var, block3_bn_relu)
    block3_ConvParam = getConv2DParam(block3_conv_weights)
    m, _ = block3_BNParam.shape
    count16 = count16 + (m * 2)
    count1 = count1 + block3_ConvParam.size + (m*2)
    # count1 = count1 + block3_ConvParam.size
    np2txt(block3_BNParam, "block3_BNParam")
    np2txt(block3_ConvParam, "block3_ConvParam")

    block4_conv_weights_path = "./Weights/block4_conv_tensor.npy"
    block4_conv_weights = np.load(block4_conv_weights_path)
    block4_bn_weights_path = "./Weights/block4_bn_weight_tensor.npy"
    block4_bn_weights = np.load(block4_bn_weights_path)
    block4_bn_bias_path = "./Weights/block4_bn_bias_tensor.npy"
    block4_bn_bias = np.load(block4_bn_bias_path)
    block4_bn_mean_path = "./Weights/block4_bn_mean_tensor.npy"
    block4_bn_mean = np.load(block4_bn_mean_path)
    block4_bn_var_path = "./Weights/block4_bn_var_tensor.npy"
    block4_bn_var = np.load(block4_bn_var_path)
    block4_bn_relu_path = "./Weights/block4_prelu_tensor.npy"
    block4_bn_relu = np.load(block4_bn_relu_path)
    block4_BNParam = getBNParam(block4_bn_weights, block4_bn_bias, block4_bn_mean, block4_bn_var, block4_bn_relu)
    block4_ConvParam = getConv2DParam(block4_conv_weights)
    m, _ = block4_BNParam.shape
    count16 = count16 + (m * 2)
    count1 = count1 + block4_ConvParam.size + (m*2)
    # count1 = count1 + block4_ConvParam.size
    np2txt(block4_BNParam, "block4_BNParam")
    np2txt(block4_ConvParam, "block4_ConvParam")

    block5_conv_weights_path = "./Weights/block5_conv_tensor.npy"
    block5_conv_weights = np.load(block5_conv_weights_path)
    block5_bn_weights_path = "./Weights/block5_bn_weight_tensor.npy"
    block5_bn_weights = np.load(block5_bn_weights_path)
    block5_bn_bias_path = "./Weights/block5_bn_bias_tensor.npy"
    block5_bn_bias = np.load(block5_bn_bias_path)
    block5_bn_mean_path = "./Weights/block5_bn_mean_tensor.npy"
    block5_bn_mean = np.load(block5_bn_mean_path)
    block5_bn_var_path = "./Weights/block5_bn_var_tensor.npy"
    block5_bn_var = np.load(block5_bn_var_path)
    block5_bn_relu_path = "./Weights/block5_prelu_tensor.npy"
    block5_bn_relu = np.load(block5_bn_relu_path)
    block5_BNParam = getBNParam(block5_bn_weights, block5_bn_bias, block5_bn_mean, block5_bn_var, block5_bn_relu)
    block5_ConvParam = getConv1DParam(block5_conv_weights)
    m, _ = block5_BNParam.shape
    count16 = count16 + (m * 2)
    count1 = count1 + block5_ConvParam.size + (m*2)
    # count1 = count1 + block5_ConvParam.size
    np2txt(block5_BNParam, "block5_BNParam")
    np2txt(block5_ConvParam, "block5_ConvParam")

    block6_conv_weights_path = "./Weights/block6_conv_tensor.npy"
    block6_conv_weights = np.load(block6_conv_weights_path)
    block6_bn_weights_path = "./Weights/block6_bn_weight_tensor.npy"
    block6_bn_weights = np.load(block6_bn_weights_path)
    block6_bn_bias_path = "./Weights/block6_bn_bias_tensor.npy"
    block6_bn_bias = np.load(block6_bn_bias_path)
    block6_bn_mean_path = "./Weights/block6_bn_mean_tensor.npy"
    block6_bn_mean = np.load(block6_bn_mean_path)
    block6_bn_var_path = "./Weights/block6_bn_var_tensor.npy"
    block6_bn_var = np.load(block6_bn_var_path)
    block6_bn_relu_path = "./Weights/block6_prelu_tensor.npy"
    block6_bn_relu = np.load(block6_bn_relu_path)
    block6_BNParam = getBNParam(block6_bn_weights, block6_bn_bias, block6_bn_mean, block6_bn_var, block6_bn_relu)
    block6_ConvParam = getConv1DParam(block6_conv_weights)
    # 最后一层的BN实际上是无效的
    count1 = count1 + block6_ConvParam.size
    np2txt(block6_BNParam, "block6_BNParam")
    np2txt(block6_ConvParam, "block6_ConvParam")

    # print("1位宽参数数量为: ", count1)
    # print("16位宽参数数量为: ", count16)
    # print("实际占据字节长度: ", (((count1/8)+(count16*2))/1024), " KB")


def detail_sim(image_iact):
    t = 0
    block1_conv_weights_path = "./Weights/128-origin/block1_conv_tensor.npy"
    block1_conv_weights = np.load(block1_conv_weights_path)
    block1_bn_weights_path = "./Weights/128-origin/block1_bn_weight_tensor.npy"
    block1_bn_weights = np.load(block1_bn_weights_path)
    block1_bn_bias_path = "./Weights/128-origin/block1_bn_bias_tensor.npy"
    block1_bn_bias = np.load(block1_bn_bias_path)
    block1_bn_mean_path = "./Weights/128-origin/block1_bn_mean_tensor.npy"
    block1_bn_mean = np.load(block1_bn_mean_path)
    block1_bn_var_path = "./Weights/128-origin/block1_bn_var_tensor.npy"
    block1_bn_var = np.load(block1_bn_var_path)
    block1_bn_relu_path = "./Weights/128-origin/block1_prelu_tensor.npy"
    block1_bn_relu = np.load(block1_bn_relu_path)

    # shape = block1_conv_weights.shape
    # total_params = np.prod(shape)
    # t = t + total_params

    block1_result = block_cal_2d(image_iact, block1_conv_weights, block1_bn_weights, block1_bn_bias, block1_bn_mean,
                                 block1_bn_var, block1_bn_relu, 2, 1)

    print("block1: ", np.prod(block1_result.shape))

    block2_conv_weights_path = "./Weights/128-origin/block2_conv_tensor.npy"
    block2_conv_weights = np.load(block2_conv_weights_path)
    block2_bn_weights_path = "./Weights/128-origin/block2_bn_weight_tensor.npy"
    block2_bn_weights = np.load(block2_bn_weights_path)
    block2_bn_bias_path = "./Weights/128-origin/block2_bn_bias_tensor.npy"
    block2_bn_bias = np.load(block2_bn_bias_path)
    block2_bn_mean_path = "./Weights/128-origin/block2_bn_mean_tensor.npy"
    block2_bn_mean = np.load(block2_bn_mean_path)
    block2_bn_var_path = "./Weights/128-origin/block2_bn_var_tensor.npy"
    block2_bn_var = np.load(block2_bn_var_path)
    block2_bn_relu_path = "./Weights/128-origin/block2_prelu_tensor.npy"
    block2_bn_relu = np.load(block2_bn_relu_path)
    # shape = block2_conv_weights.shape
    # total_params = np.prod(shape)
    # t = t + total_params
    block2_result = block_cal_2d(block1_result, block2_conv_weights, block2_bn_weights, block2_bn_bias, block2_bn_mean,
                                 block2_bn_var, block2_bn_relu, 2, 2)
    print("block1: ", np.prod(block2_result.shape))

    block3_conv_weights_path = "./Weights/128-origin/block3_conv_tensor.npy"
    block3_conv_weights = np.load(block3_conv_weights_path)
    block3_bn_weights_path = "./Weights/128-origin/block3_bn_weight_tensor.npy"
    block3_bn_weights = np.load(block3_bn_weights_path)
    block3_bn_bias_path = "./Weights/128-origin/block3_bn_bias_tensor.npy"
    block3_bn_bias = np.load(block3_bn_bias_path)
    block3_bn_mean_path = "./Weights/128-origin/block3_bn_mean_tensor.npy"
    block3_bn_mean = np.load(block3_bn_mean_path)
    block3_bn_var_path = "./Weights/128-origin/block3_bn_var_tensor.npy"
    block3_bn_var = np.load(block3_bn_var_path)
    block3_bn_relu_path = "./Weights/128-origin/block3_prelu_tensor.npy"
    block3_bn_relu = np.load(block3_bn_relu_path)
    # shape = block3_conv_weights.shape
    # total_params = np.prod(shape)
    # t = t + total_params
    block3_result = block_cal_2d(block2_result, block3_conv_weights, block3_bn_weights, block3_bn_bias, block3_bn_mean,
                                 block3_bn_var, block3_bn_relu, 1, 3)

    print("block2: ", np.prod(block3_result.shape))

    block4_conv_weights_path = "./Weights/128-origin/block4_conv_tensor.npy"
    block4_conv_weights = np.load(block4_conv_weights_path)
    block4_bn_weights_path = "./Weights/128-origin/block4_bn_weight_tensor.npy"
    block4_bn_weights = np.load(block4_bn_weights_path)
    block4_bn_bias_path = "./Weights/128-origin/block4_bn_bias_tensor.npy"
    block4_bn_bias = np.load(block4_bn_bias_path)
    block4_bn_mean_path = "./Weights/128-origin/block4_bn_mean_tensor.npy"
    block4_bn_mean = np.load(block4_bn_mean_path)
    block4_bn_var_path = "./Weights/128-origin/block4_bn_var_tensor.npy"
    block4_bn_var = np.load(block4_bn_var_path)
    block4_bn_relu_path = "./Weights/128-origin/block4_prelu_tensor.npy"
    block4_bn_relu = np.load(block4_bn_relu_path)
    # shape = block4_conv_weights.shape
    # total_params = np.prod(shape)
    # t = t + total_params
    block4_result = block_cal_2d(block3_result, block4_conv_weights, block4_bn_weights, block4_bn_bias, block4_bn_mean,
                                 block4_bn_var, block4_bn_relu, 1, 4)
    print("block4: ", np.prod(block4_result.shape))

    # 把数据拉平，方便后面1D卷积
    block4_result = block4_result.reshape(64, -1)
    # print(block4_result.shape)

    block5_conv_weights_path = "./Weights/128-origin/block5_conv_tensor.npy"
    block5_conv_weights = np.load(block5_conv_weights_path)
    block5_bn_weights_path = "./Weights/128-origin/block5_bn_weight_tensor.npy"
    block5_bn_weights = np.load(block5_bn_weights_path)
    block5_bn_bias_path = "./Weights/128-origin/block5_bn_bias_tensor.npy"
    block5_bn_bias = np.load(block5_bn_bias_path)
    block5_bn_mean_path = "./Weights/128-origin/block5_bn_mean_tensor.npy"
    block5_bn_mean = np.load(block5_bn_mean_path)
    block5_bn_var_path = "./Weights/128-origin/block5_bn_var_tensor.npy"
    block5_bn_var = np.load(block5_bn_var_path)
    block5_bn_relu_path = "./Weights/128-origin/block5_prelu_tensor.npy"
    block5_bn_relu = np.load(block5_bn_relu_path)
    # shape = block5_conv_weights.shape
    # total_params = np.prod(shape)
    # t = t + total_params
    block5_result = block_cal_1d(block4_result, block5_conv_weights, block5_bn_weights, block5_bn_bias, block5_bn_mean,
                                 block5_bn_var, block5_bn_relu, 2, False)

    print("block5: ", np.prod(block5_result.shape))

    block6_conv_weights_path = "./Weights/128-origin/block6_conv_tensor.npy"
    block6_conv_weights = np.load(block6_conv_weights_path)
    block6_bn_weights_path = "./Weights/128-origin/block6_bn_weight_tensor.npy"
    block6_bn_weights = np.load(block6_bn_weights_path)
    block6_bn_bias_path = "./Weights/128-origin/block6_bn_bias_tensor.npy"
    block6_bn_bias = np.load(block6_bn_bias_path)
    block6_bn_mean_path = "./Weights/128-origin/block6_bn_mean_tensor.npy"
    block6_bn_mean = np.load(block6_bn_mean_path)
    block6_bn_var_path = "./Weights/128-origin/block6_bn_var_tensor.npy"
    block6_bn_var = np.load(block6_bn_var_path)
    block6_bn_relu_path = "./Weights/128-origin/block6_prelu_tensor.npy"
    block6_bn_relu = np.load(block6_bn_relu_path)
    # shape = block6_conv_weights.shape
    # total_params = np.prod(shape)
    # t = t + total_params
    block6_result = block_cal_1d(block5_result, block6_conv_weights, block6_bn_weights, block6_bn_bias, block6_bn_mean,
                                 block6_bn_var, block6_bn_relu, 2, True)
    #
    final_result = block6_result.sum(axis=1)
    # print("final result: ", final_result.shape)
    print("卷积参数的个数: ", t)
    final_category = np.argmax(final_result)
    return final_category


def image_to_binary(image):
    # 确保图像是NumPy数组
    image = np.array(image)
    # 确保图像是三维的
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3-dimensional array")
    # 将RGB图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 应用大津法进行二值化，自动确定阈值
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_image


def testOne(name):
    # 获取到测试集，用来评估模型的准确性
    # name = "00458.png"
    data_dir = './GTSRB/'
    test_path = './GTSRB/Test/'

    Y_test = pd.read_csv(data_dir + 'Test.csv')
    img_path = test_path + name
    # 加载图像
    # 加载图像
    image = Image.open(img_path)
    # 将图像调整大小
    image = image.resize((128, 128))
    image = image_to_binary(image)
    image = image / 255
    zero_positions = image == 0
    # 将这些位置的值设置为-1
    # 这样会方便部署
    image[zero_positions] = -1
    image = np.expand_dims(image, axis=0)

    np_array = detail_sim(image)
    # np_array = np_array.astype(np.float32)
    answer = Y_test.loc[Y_test['ImageId'] == name, 'ClassId'].values[0]
    r = str("实际分类: " + str(answer) + " -- 预测分类: " + str(np_array))
    print(r)
    return r


from utils.model import ECG_XNOR_Img


def loadWeightFromBestModel():
    # 加载模型
    loaded = torch.load('best_model.pth', weights_only=False)

    # 检查加载的对象类型
    if isinstance(loaded, dict):  # 如果是字典，直接使用
        state_dict = loaded
    else:  # 如果是模型类实例，尝试获取其状态字典
        state_dict = loaded.state_dict()

    device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    kernel_size, padding, poolsize, kernel_size_1d, poolsize_1d = 3, 0, 2, 7, 4
    padding_value = 0
    A = [[1, 8, kernel_size, 1, padding, padding_value, poolsize, 2],
         [8, 16, kernel_size, 1, padding, padding_value, poolsize, 2],
         [16, 32, kernel_size, 1, padding, padding_value, poolsize, 1],
         [32, 64, kernel_size, 1, padding, padding_value, poolsize, 1],
         [64, 128, kernel_size_1d, 1, padding, padding_value, poolsize_1d, 2, False],
         [128, 43, kernel_size_1d, 1, padding, padding_value, poolsize_1d, 2, True]
         ]

    model = ECG_XNOR_Img(block1=A[0], block2=A[1], block3=A[2], block4=A[3],
                         block5=A[4] if len(A) > 4 else None,
                         block6=A[5] if len(A) > 5 else None,
                         block7=A[6] if len(A) > 6 else None,
                         device=device_cpu).to(device_cpu)

    # 加载状态字典
    # model.load_state_dict(state_dict)

    weights_dict = model.state_dict()
    for key in weights_dict:
        # 获取权重的NumPy表示
        np_weights = weights_dict[key].numpy()
        # 保存为.npy文件
        np.save(f'./Weights/{key}.npy', np_weights)

# def test_best_model(input):
#     # 加载模型
#     loaded = torch.load('best_model.pth', weights_only=False)
#
#     # 检查加载的对象类型
#     if isinstance(loaded, dict):  # 如果是字典，直接使用
#         state_dict = loaded
#     else:  # 如果是模型类实例，尝试获取其状态字典
#         state_dict = loaded.state_dict()
#
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     kernel_size, padding, poolsize, kernel_size_1d, poolsize_1d = 3, 0, 2, 7, 4
#     padding_value = 0
#     A = [[1, 8, kernel_size, 1, padding, padding_value, poolsize, 2],
#          [8, 16, kernel_size, 1, padding, padding_value, poolsize, 2],
#          [16, 32, kernel_size, 1, padding, padding_value, poolsize, 1],
#          [32, 64, kernel_size, 1, padding, padding_value, poolsize, 1],
#          [64, 128, kernel_size_1d, 1, padding, padding_value, poolsize_1d, 2, False],
#          [128, 43, kernel_size_1d, 1, padding, padding_value, poolsize_1d, 2, True]
#          ]
#
#     model = ECG_XNOR_Img(block1=A[0], block2=A[1], block3=A[2], block4=A[3],
#                          block5=A[4] if len(A) > 4 else None,
#                          block6=A[5] if len(A) > 5 else None,
#                          block7=A[6] if len(A) > 6 else None,
#                          device=device).to(device)
#
#     # 加载状态字典
#     model.load_state_dict(state_dict)
#
#     # 设置为评估模式
#     model.eval()
#
#     input_tensor = torch.tensor(input).float()
#     # 进行推理
#     with torch.no_grad():
#         output = model(input_tensor)
#
#     #return int(output.argmax(dim=1))
#     return output


def testAll():
    # 获取到测试集，用来评估模型的准确性
    data_dir = './GTSRB/'
    train_path = './GTSRB/Train/'
    test_path = './GTSRB/Test/'

    Y_test = pd.read_csv(data_dir + 'Test.csv')
    test_labels = Y_test["ClassId"].values
    test_images = Y_test["ImageId"].values

    imgs = list()
    count = 0
    for img in test_images:
        if count < 100:
            # 加载图像
            image = Image.open(os.path.join(test_path, img))
            # 将图像调整大小
            image = image.resize((128, 128))
            image = image_to_binary(image)
            image = image / 255
            zero_positions = image == 0
            # 将这些位置的值设置为-1
            # 这样会方便部署
            image[zero_positions] = -1
            image = np.expand_dims(image, axis=0)
            imgs.append(np.array(image))
            count = count + 1

    test_images = np.array(imgs)
    test_labels = np.array(test_labels)

    count = 0
    num, n, _, ch = test_images.shape
    # 使用 tqdm 包装你的迭代器 -- 方便展示进度条
    for i in tqdm(range(num), desc='Processing images'):
        input = test_images[i]
        # predict = sim(input, False)
        predict = detail_sim(input)
        # 预测结果和实际结果对应，成功数据+1
        if predict == test_labels[i]:
            count += 1

    print("max: ", max, " --min: ", min)
    print(f'Accuracy: {count / num * 100:.2f}%')


image = np.zeros((1, 64, 64))

# testAll()
# get_weights()
import time

start_time = time.perf_counter()  # 记录开始时间

# === 你的方法 ===
testOne("00854.png")
# ===============

end_time = time.perf_counter()    # 记录结束时间

# 计算耗时（秒 -> 毫秒）
latency_ms = (end_time - start_time) * 1000

print(f"⏱️  推理耗时: {latency_ms:.2f} ms") # 保留两位小数
#loadWeightFromBestModel()
# print("运算总次数：", conv_ops_count)
