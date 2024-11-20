import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_difference_medical_images(image_path1, image_path2):
    # 加载医学图像
    image1 = sitk.ReadImage(image_path1)
    image2 = sitk.ReadImage(image_path2)

    # 转换为NumPy数组
    array1 = sitk.GetArrayFromImage(image1)
    array2 = sitk.GetArrayFromImage(image2)

    # 计算差值图像
    diff_image = np.abs(array1 - array2)

    return diff_image


# 替换为你的实际医学图像文件路径
image_path1 = "/group/xuyawen/Attention-Reg-main_backup/figs/1701788914.nii.gz"
image_path2 = "/group/xuyawen/Attention-Reg-main_backup/figs/1701788915.nii.gz"
image1 = sitk.ReadImage(image_path1)
image2 = sitk.ReadImage(image_path2)
# 转换为NumPy数组
array1 = sitk.GetArrayFromImage(image1)
array2 = sitk.GetArrayFromImage(image2)
# 计算差值图像
difference_image = compute_difference_medical_images(image_path1, image_path2)
plt.imshow(array1[31,:,:],cmap='gray')
plt.title('image_path1')
plt.show()

plt.imshow(array2[31,:,:], cmap='gray')
plt.title('image_path2')
plt.show()
# 可视化差值图像的某一层


plt.imshow(difference_image[31,:,:], cmap='gray')
plt.title('Difference Image')
plt.colorbar()
plt.show()
