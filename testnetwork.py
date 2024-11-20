import torch
import numpy as np
import os
import SimpleITK as sitk
from os import path
import cv2
from torch import nn
import metric as mt
from metric import ssim3d
from utils import data_loading_funcs as load_func
import cross_attention as gens
import math
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
import matplotlib.pyplot as plt
import time


def ssim(gray1, gray2):
    # 定义 SSIM 相关参数
    K1 = 0.01
    K2 = 0.03
    L = 255

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    gray1 = np.mean(gray1, axis=0)
    gray2 = np.mean(gray2, axis=0)
    # 均值
    mu1 = convolve2d(gray1, np.ones((8, 8)) / 64.0, mode='same', boundary='symm')
    mu2 = convolve2d(gray2, np.ones((8, 8)) / 64.0, mode='same', boundary='symm')

    # 方差
    sigma1_sq = gaussian_filter(gray1 ** 2, sigma=1) - mu1 ** 2
    sigma2_sq = gaussian_filter(gray2 ** 2, sigma=1) - mu2 ** 2

    # 协方差
    sigma12 = gaussian_filter(gray1 * gray2, sigma=1) - mu1 * mu2

    # SSIM 公式
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    # 返回平均 SSIM
    return np.mean(ssim_map)

def ncc(volume1, volume2):
    # 计算均值
    mean_volume1 = np.mean(volume1)
    mean_volume2 = np.mean(volume2)

    # 计算归一化互相关系数的分子部分
    numerator = np.sum((volume1 - mean_volume1) * (volume2 - mean_volume2))

    # 计算归一化互相关系数的分母部分
    denominator = np.sqrt(np.sum((volume1 - mean_volume1)**2) * np.sum((volume2 - mean_volume2)**2))

    # 计算归一化互相关系数
    ncc_value = numerator / denominator

    return ncc_value

def mutual_information(volume1, volume2):
    # 将体数据展平为一维数组
    flat_volume1 = volume1.flatten()
    flat_volume2 = volume2.flatten()

    # 计算联合概率分布
    joint_distribution = np.histogram2d(flat_volume1, flat_volume2, bins=20)[0]

    # 计算边缘概率分布
    marginal_distribution_volume1 = np.histogram(flat_volume1, bins=20)[0]
    marginal_distribution_volume2 = np.histogram(flat_volume2, bins=20)[0]

    # 计算互信息
    mi_value = entropy(joint_distribution.flatten()) - entropy(marginal_distribution_volume1) - entropy(marginal_distribution_volume2)

    return mi_value

"""load test data"""
def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    # print('min of scaled {}'.format(np.min(scaled_volume)))
    # print('max of scaled {}'.format(np.max(scaled_volume)))
    return scaled_volume
def normalize_volume(input_volume):
    # print('input_volume shape {}'.format(input_volume.shape))
    mean = np.mean(input_volume)
    std = np.std(input_volume)
    normalized_volume = (input_volume - mean) / std
    return normalized_volume



def absu(inputs,outputs,targets):


    input_image = torch.squeeze(inputs).cpu().numpy()
    input_image = input_image[1, :, :, :]
    input_image = sitk.GetImageFromArray(input_image)


    translation_x = float(outputs[0])
    translation_y = float(outputs[1])
    translation_z = float(outputs[2])
    translation_params =[translation_x, translation_y, translation_z]
    translation_params = np.array(translation_params)  #搞了一下午 必须是常量 python没有常量,nonono 不是数据结构 是数据库类型问题
    translation_transform = sitk.TranslationTransform(3)
    translation_transform.SetOffset(translation_params)

    rotation_x = float(outputs[3])
    rotation_y = float(outputs[4])
    rotation_z = float(outputs[5])
    rotation_params = [rotation_x,  rotation_y,  rotation_z]
    rotation_params = np.array(rotation_params)  # 搞了一下午 必须是常量 python没有常量
    rotation_transform = sitk.Euler3DTransform()
    rotation_transform.SetRotation(rotation_params[0], rotation_params[1], rotation_params[2])

    composite_out_transform = sitk.CompositeTransform(translation_transform)
    composite_out_transform.AddTransform(rotation_transform)

    output_all_image = sitk.Resample(input_image, composite_out_transform)

    translation_x = float(targets[0])
    translation_y = float(targets[1])
    translation_z = float(targets[2])
    translation_params = [translation_x, translation_y, translation_z]
    translation_params = np.array(translation_params)  # 搞了一下午 必须是常量 python没有常量
    translation_transform = sitk.TranslationTransform(3)
    translation_transform.SetOffset(translation_params)

    rotation_x = float(targets[3])
    rotation_y = float(targets[4])
    rotation_z = float(targets[5])
    rotation_params = [rotation_x, rotation_y, rotation_z]
    rotation_params = np.array(rotation_params)
    rotation_transform = sitk.Euler3DTransform()
    rotation_transform.SetRotation(rotation_params[0], rotation_params[1], rotation_params[2])

    composite_tar_transform = sitk.CompositeTransform(translation_transform)
    composite_tar_transform.AddTransform(rotation_transform)

    # target_tran_image = sitk.Resample(input_image, translation_transform)
    # target_rota_image = sitk.Resample(input_image, rotation_transform)
    target_all_image = sitk.Resample(input_image, composite_tar_transform)

    error = np.mean(np.abs(target_all_image - output_all_image))
    error_tensor = torch.tensor(error, requires_grad=True)
    array1 = sitk.GetArrayFromImage(target_all_image)
    array2 = sitk.GetArrayFromImage(output_all_image)
    timestamp = int(time.time())
    output_path1 = "figs/{}.nii.gz".format(timestamp+1)
    output_path = "figs/{}.nii.gz".format(timestamp)

    sitk.WriteImage(output_all_image, output_path1)
    sitk.WriteImage(target_all_image, output_path)

    # plt.imshow(array1[10, : , :])
    # plt.show()
    # plt.imshow(array2[10, : , :])
    # plt.show()
    # src = mlab.pipeline.scalar_field(array1)
    # # 创建三维图
    # mlab.contour3d(src)
    #
    # # 显示图形
    # mlab.show()
    ss= ssim(array1, array2)

    # print("ssim",ss)
    return ss

if __name__ == '__main__':

    i = 0

    error_mean_tx = []
    error_mean_ty = []
    error_mean_tz = []
    error_mean_rx = []
    error_mean_ry = []
    error_mean_rz = []
    ncc_mean = []
    ssim_mean = []

    test_dir = "split/test"


    """load trained model"""
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    att_gen = gens.AttentionReg()
    # att_gen = nn.DataParallel(att_gen)
    """*************************MODEL NAME HERE********************************"""
    att_gen.load_state_dict(torch.load(
                    "/home/jiwenxiu/mapping/results/cross_attention_AttentionReg_1128-101853_load_model.pth")) #MFF


    device=torch.device('cuda', 1)
    att_gen.eval()
    att_gen.cuda().to(device)
    # gt_trans_fn = path.join('sample', 'gt.txt')
    gt_trans_fn = path.join('us_trans_final/gt.txt')

    gt_mat = np.loadtxt(gt_trans_fn)


    for filename in os.listdir(test_dir):
        img_file = os.path.join(test_dir, filename)
        print(img_file)
        sample4D = np.zeros((2, 32, 96, 96), dtype=np.ubyte)
        for file in os.listdir(img_file):
            # print(file)
            if file.endswith('.txt'):
                base_mat = np.loadtxt(path.join(img_file, file))
            if file.endswith('.npy') and file[:2] == "CT":
                sample4D[0, :, :, :] = np.load(path.join(img_file,file))
            if file.endswith('.npy') and file[:2] == "US":
                sample4D[1, :, :, :] = np.load(path.join(img_file,file))

        # sample4D = scale_volume(sample4D,1,0)
        sample4D = normalize_volume(sample4D)
        mat_diff = gt_mat.dot(np.linalg.inv(base_mat))
        target = load_func.decompose_matrix_degree(mat_diff)

        inputs = np.expand_dims(sample4D, axis=0)

        """feeding test data to the network"""
        inputs = torch.from_numpy(inputs).float().to(device)
        outputs = att_gen(inputs)
        outputs = outputs.data.cpu().numpy().flatten()
        s= absu(inputs, outputs, target)
        # add_params = np.reshape(outputs, (outputs.shape[1]))

        """evaluation"""
         # registration_mat = load_func.construct_matrix_degree(params=add_params, initial_transform=base_mat)
        mse_tx = abs(outputs[0] - target[0])
        mse_ty = abs(outputs[1] - target[1])
        mse_tz = abs(outputs[2] - target[2])
        mse_rx = abs(outputs[3] - target[3])
        mse_ry = abs(outputs[4] - target[4])
        mse_rz = abs(outputs[5] - target[5])

        error_mean_tx.append(mse_tx)
        error_mean_ty.append(mse_ty)
        error_mean_tz.append(mse_tz)
        error_mean_rx.append(mse_rx)
        error_mean_ry.append(mse_ry)
        error_mean_rz.append(mse_rz)

        # print(outputs)
        # print(target)
        # print('testing MSE-T error= {}'.format(mse_tx))
        # print('testing MSE_r error= {}'.format(mse_rx))
        # result = cross_correlation(data1, data2)
        ssim_mean.append(s)


    print("mse_tx平均为：", np.mean(error_mean_tx))
    print("mse_tx 方差：", np.var(error_mean_tx))
    print("---------------------------------------")

    print("mse_ty平均为：", np.mean(error_mean_ty))
    print("mse_ty 方差：", np.var(error_mean_ty))
    print("---------------------------------------")
    print("mse_tz平均为：", np.mean(error_mean_tz))
    print("mse_tz 方差：", np.var(error_mean_tz))
    print("---------------------------------------")
    print("mse_rx平均为：", np.mean(error_mean_rx))
    print("mse_rx 方差：", np.var(error_mean_rx))
    print("---------------------------------------")
    print("mse_ry平均为：", np.mean(error_mean_ry))
    print("mse_ry 方差：", np.var(error_mean_ry))
    print("---------------------------------------")
    print("mse_rz平均为：", np.mean(error_mean_rz))
    print("mse_rz方差：", np.var(error_mean_rz))
    print("---------------------------------------")

    print("ssim平均为：", np.mean(ssim_mean))
    print("ssim 方差：", np.var(ssim_mean))
    # print("ssim：", ssim_mean)



