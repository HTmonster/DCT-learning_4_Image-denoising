# 基于字典学习的图像去噪研究

#### 目录介绍

- image: 存放原图和处理后的图像
- ImageTools.py 图像辅助处理函数集合

#### :chart_with_upwards_trend: 进度记录

- 图像工具
    - 图像加噪
        - [x]  高斯噪声
        - [x]  椒盐噪声
        - [x]  瑞利噪声
        - [x]  伽马噪声
        - [x]  均匀噪声
    - 去噪效果评价标准
        - [x]  SNR 信噪比
        - [x]  PSNR 峰值信噪比
        - [x]  SSIM 结构相似度
- 字典学习
    - [x] KSVD字典学习
    - [x] OMP稀疏表示

## 去噪初步效果
![初步效果](.\images\denoise\KSVD_n64_iter30_tol1e-06.png)