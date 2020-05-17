
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
    - [x] 正则化KSVD
#### 加噪效果展示

| 噪声类型 | 效果                                                         |
| -------- | ------------------------------------------------------------ |
| 高斯噪声 | <img src="images\noise\SaltPepperNoise_per0.03_modeBOTH_lenna.jpg" style="zoom:25%;" /> |
| 椒盐噪声 | <img src="images\noise\SaltPepperNoise_per0.08_modeBOTH_lenna.jpg" style="zoom:25%;" /> |
| 伽马噪声 | <img src="images\noise\GammaNoise_k30_theta4_per0.5_lenna.jpg" style="zoom:25%;" /> |
| 瑞利噪声 | <img src="images\noise\RayleighNoise_per0.5_scale60_lenna.jpg" style="zoom:25%;" /> |
| 指数噪声 | <img src="images\noise\Exponential_beta50_per0.5_lenna.jpg" style="zoom:25%;" /> |
| 均匀噪声 | <img src="images\noise\UniformNoise_a50_b150_per0.5_lenna.jpg" style="zoom:25%;" /> |

#### 去噪效果展示
![去噪效果](.\images\denoise\regKSVD_n50_iter200_12_10_25.png)

