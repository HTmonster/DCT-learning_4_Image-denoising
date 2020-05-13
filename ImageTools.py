
#==============================================
#           ImageTools.py 
#       图像处理辅助函数集合
#
# ****************图像噪声分类 ******************
# 
#   1.加性噪声  f(x,y)=g(x,y)+n(x,y) 
#       一般是图像传输信道噪声和CCD摄像机图像数字化过程中产生
#   2.乘性噪声 f(x,y)=g(x,y)*n(x,y)
#       一般由 胶片中颗粒 飞点扫描图像噪声 电视扫描光栅等原因造成
#   3.量化噪声
#       模拟到数字产生的差异 量化中的误差

# *****************图像噪声模型******************
# 
#    1.高斯噪声 (Gaussian noise)
#       最广泛。传感器非正常环境下产生，电子电路中噪声。 高斯分布
#    2.脉冲噪声 (Impulsive noise)
#       双极脉冲：椒盐脉冲，尖峰噪声 散粒噪声 
#                   盐噪声：随机的白色像素点
#                   胡椒噪声：随机黑色像素点
#    3.瑞利噪声 (Rayleigh noise)
#    4.伽马(爱尔兰)噪声 (Gamma noise)
#    5.指数噪声(Exponential noise)
#    6.均匀噪声(Uniform noise)
#
# *****************去噪效果评价算法***************
#    1.SNR  [信噪比] 计算图像自身的信噪比  输入为一幅图片
#    2.PSNR [峰值信噪比] 计算两个图像之间的相似度  去噪后的图片和原图做比较
#    3.SSIM [结构相似性] 衡量两幅图像相似度
#=================================================
import numpy as np
import copy
import cv2
import random
import skimage.metrics

def GaussianNoise(srcImg,percent,sigma,means=0,greyscale=256):   
    """
    为灰度图像添加 高斯噪声
        :param srcImg: 源图像
        :param percent: 噪声百分比
        :param sigma: 高斯的标准差
        :param means=0: 高斯的均值  默认为0
        :param greyscale=256: 灰度图像的度  默认为256
    """
    (h,w)= srcImg.shape     #源图像的长宽
    NoiseImg=copy.deepcopy(srcImg)           #复制源图像
    places=random.sample(range(h*w),int(percent*w*h))            #随机按照百分比选择噪声点位置

    for place in places:
        x,y=place%w,place//w # 坐标
        # f(x,y)=g(x,y)+n(x,y) 
        fxy=int(NoiseImg[x,y]+random.gauss(means,sigma))
        # 范围修正
        if fxy<0:
            fxy=0
        elif fxy>greyscale-1:
            fxy=greyscale-1
        NoiseImg[x,y]=fxy
    return NoiseImg

def SaltPepperNoise(srcImg,percent,mode="BOTH",greyscale=256):
    """
    对灰度图像添加椒盐噪声
        :param srcImg: 源图片
        :param percent: 百分比
        :param mode="BOTH": 模式：BOTH:椒盐模式  SALT:盐 PEPPRR 椒
        :param greyscale=256: 灰度图像的度  默认为256
    """
    (h,w)= srcImg.shape     #源图像的长宽
    NoiseImg=copy.deepcopy(srcImg)           #复制源图像
    places=random.sample(range(h*w),int(percent*w*h))            #随机按照百分比选择噪声点位置

    for place in places:
        x,y=place%w,place//w # 坐标
        if mode=="SALT":     #加盐
            NoiseImg[x,y]=255
        elif mode=="PEPPER": #加胡椒
            NoiseImg[x,y]=0
        else:                #
            NoiseImg[x,y]=0 if random.random()<=0.5 else 255
    return NoiseImg

def RayleighNoise(srcImg,scale,percent,greyscale=256):
    """
    对灰度头像加上瑞利噪声
        :param srcImg: 源图像
        :param scale: 规模
        :param percent:百分比
        :param greyscale=256: 灰度图像的度  默认为256
    """

    (h,w)= srcImg.shape     #源图像的长宽
    NoiseImg=copy.deepcopy(srcImg)           #复制源图像

    places=random.sample(range(h*w),int(percent*w*h))    #位置
    Rayleighs=np.random.rayleigh(scale,int(percent*w*h)) #瑞利噪声

    for i in range(int(percent*w*h)):
        x,y=places[i]%w,places[i]//w # 坐标
        fxy=NoiseImg[x,y]+int(Rayleighs[i]) #加上噪声
         # 范围修正
        if fxy<0:
            fxy=0
        elif fxy>greyscale-1:
            fxy=greyscale-1
        NoiseImg[x,y]=fxy
    return NoiseImg

def GammaNoise(srcImg,k,theta,percent,greyscale=256):
    """
    为灰度图像添加伽马噪声
        :param srcImg: 源图像
        :param k: 伽马分布参数k
        :param theta: 伽马分布参数theta
        :param percent: 百分比
        :param greyscale=256:  灰度图像的度  默认为256
    """

    (h,w)= srcImg.shape     #源图像的长宽
    NoiseImg=copy.deepcopy(srcImg)           #复制源图像

    places=random.sample(range(h*w),int(percent*w*h))    #位置
    Gammas=np.random.gamma(k,theta,int(percent*w*h)) #伽马噪声

    for i in range(int(percent*w*h)):
        x,y=places[i]%w,places[i]//w # 坐标
        fxy=NoiseImg[x,y]+int(Gammas[i]) #加上噪声
         # 范围修正
        if fxy<0:
            fxy=0
        elif fxy>greyscale-1:
            fxy=greyscale-1
        NoiseImg[x,y]=fxy
    return NoiseImg


def ExponentialNoise(srcImg,beta,percent,greyscale=256):
    """
    为灰度图像添加指数噪声
        :param srcImg: 源图像
        :param beta: 指数分布参数
        :param percent: 百分比
        :param greyscale=256:  灰度图像的度  默认为256
    """

    (h,w)= srcImg.shape     #源图像的长宽
    NoiseImg=copy.deepcopy(srcImg)           #复制源图像

    places=random.sample(range(h*w),int(percent*w*h))    #位置
    Expons=np.random.exponential(beta,int(percent*w*h))  #指数噪声

    for i in range(int(percent*w*h)):
        x,y=places[i]%w,places[i]//w # 坐标
        fxy=NoiseImg[x,y]+int(Expons[i]) #加上噪声
         # 范围修正
        if fxy<0:
            fxy=0
        elif fxy>greyscale-1:
            fxy=greyscale-1
        NoiseImg[x,y]=fxy
    return NoiseImg

def UniformNoise(srcImg,a,b,percent,greyscale=256):
    """
    为灰度图像添加均匀噪声
        :param srcImg: 源图像  
        :param a: 均匀分布参数a  low
        :param b: 均匀分布参数b  high
        :param percent: 百分比
        :param greyscale=256：灰度图像的度  默认为256
    """
    (h,w)= srcImg.shape     #源图像的长宽
    NoiseImg=copy.deepcopy(srcImg)           #复制源图像

    places=random.sample(range(h*w),int(percent*w*h))    #位置
    unis=np.random.uniform(a,b,int(percent*w*h))         #均匀噪声

    for i in range(int(percent*w*h)):
        x,y=places[i]%w,places[i]//w # 坐标
        fxy=NoiseImg[x,y]+int(unis[i]) #加上噪声
         # 范围修正
        if fxy<0:
            fxy=0
        elif fxy>greyscale-1:
            fxy=greyscale-1
        NoiseImg[x,y]=fxy
    return NoiseImg

def cal_SNR(img):
    """
    计算图像的SNR 使用方差法
        :param img: 
    """
    snr=np.mean(img)/np.std(img)

    return snr

def cal_PSNR(img,img_n):
    """
    计算图像的PSNR  使用skimage 自带的方法
        :param img: 原始图像
        :param img_n: 噪声图像
    """
    return skimage.metrics.peak_signal_noise_ratio(img, img_n, data_range=255)

def cal_SSIM(img,img_n):
    """
    计算图像的SSIM  使用skimage 自带的方法
        :param img: 原始图像
        :param img_n: 噪声图像
    """
    return skimage.metrics.structural_similarity(img, img_n, data_range=255)

if __name__ == "__main__":
    img=cv2.imread('.\images\original\lenna.bmp',0) 
    # cv2.imshow("original",img)

    #高斯噪声
    g_per,means,sigma=1.0,0,50
    gaussImg=GaussianNoise(img,g_per,sigma)
    cv2.imshow("guass",gaussImg)
    cv2.imwrite("./images/noise/GuassNoise_per{}_sigma{}_means{}_lenna.jpg".format(g_per,sigma,means),gaussImg)

    # 椒盐噪声
    # s_per,mode=0.08,"BOTH"
    # spImg=SaltPepperNoise(img,s_per)
    # cv2.imshow("SaltPepper",spImg)
    # cv2.imwrite("./images/noise/SaltPepperNoise_per{}_mode{}_lenna.jpg".format(s_per,mode),spImg)

    # 瑞利噪声
    # scale,r_per=60,0.5
    # rayImg=RayleighNoise(img,scale,r_per)
    # cv2.imshow("Rayleigh",rayImg)
    # cv2.imwrite("./images/noise/RayleighNoise_per{}_scale{}_lenna.jpg".format(r_per,scale),rayImg)

    # 伽马噪声
    # k,theta,ga_per=30,4,0.5
    # gammaImg=GammaNoise(img,k,theta,ga_per)
    # cv2.imshow("Gamma",gammaImg)
    # cv2.imwrite("./images/noise/GammaNoise_k{}_theta{}_per{}_lenna.jpg".format(k,theta,ga_per),gammaImg)

    # 指数噪声
    # beta,e_per=50,0.5
    # expImg=ExponentialNoise(img,beta,e_per)
    # cv2.imshow("Exponential",expImg)
    # cv2.imwrite("./images/noise/Exponential_beta{}_per{}_lenna.jpg".format(beta,e_per),expImg)
    
    # 均匀噪声
    # a,b,u_per=50,150,0.5
    # uniImg=UniformNoise(img,a,b,u_per)
    # cv2.imshow("Uniform",uniImg)
    # cv2.imwrite("./images/noise/UniformNoise_a{}_b{}_per{}_lenna.jpg".format(a,b,u_per),uniImg)
    # print(cal_SNR(img))
    # print(cal_SNR(gaussImg))

    # print(cal_PSNR(img,gaussImg))
    # print(cal_SSIM(img,gaussImg))
    
    cv2.waitKey()

