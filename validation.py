# -*- coding: utf-8 -*-
#   ____    ____    ______________
#  |    |  |    |  |              |
#  |    |  |    |  |_____    _____|
#  |    |__|    |       |    |
#  |     __     |       |    |
#  |    |  |    |       |    |
#  |    |  |    |       |    |
#  |____|  |____|       |____|
#
# Created on 2020-04-08 15:46
# @author: HTmonster
# @e-mail: Theo_hui@163.com
# 
# Des: 对去噪进行多方面的测试
from ImageTools import *
from DCTlearning import*
import cv2
import csv

def denoise_ksvd(img,n_img,n_components,iters,info=None):
    """
    使用 KSVD 来进行图像去噪
        :param img: 原图像
        :param n_img: 噪声图像
        :param n_components: 字典原子个数
        :param iters: 迭代次数
    """
    ksvd=KSVD(n_img,n_components,max_iter=iters)
    ksvd.fit()
    de_img=ksvd.denoise(addition_info=info)

    org_SNR,de_SNR=cal_SNR(n_img),cal_SNR(de_img)
    PSNR=cal_PSNR(img,de_img)
    SSIM=cal_SSIM(img,de_img)

    #返回去噪效果
    return (org_SNR,de_SNR,PSNR,SSIM)

def denoise_reg_ksvd(img,n_img,n_components,iters,info=None):    
    """
    使用 KSVD 来进行图像去噪
        :param img: 原图像
        :param n_img: 噪声图像
        :param n_components: 字典原子个数
        :param iters: 迭代次数
    """
    ksvd=RegKSVD(n_img,n_components,iters)
    ksvd.fit()
    de_img=ksvd.denoise(addition_info=info)

    org_SNR,de_SNR=cal_SNR(n_img),cal_SNR(de_img)
    PSNR=cal_PSNR(img,de_img)
    SSIM=cal_SSIM(img,de_img)

    #返回去噪效果
    return (org_SNR,de_SNR,PSNR,SSIM)

def validation_multi_noise(n_components,iters):
    ''' 
        针对一幅图 对各种噪声和各种参数 册数去噪效果
    '''
    img=cv2.imread('./images/original/baboo.bmp',0) 

    #噪声及噪声参数
    noises={
        GaussianNoise:[(1.0,5),(1.0,10),(1.0,20),(1.0,30),(1.0,50)],
        SaltPepperNoise:[(0.01,),(0.02,),(0.05,),(0.08,),(0.1,)],
        RayleighNoise:[(20,0.5),(30,0.5),(40,0.5),(50,0.5),(60,0.5)],
        GammaNoise:[(20,1,0.5),(25,1,0.5),(30,1,0.5),
                    (20,2,0.5),(25,2,0.5),(30,2,0.5),
                    (20,4,0.5),(25,4,0.5),(30,4,0.5),],
        ExponentialNoise:[(20,0.5),(50,0.5),(100,0.5)],
        UniformNoise:[(0,50,0.5),(0,100,0.5)]
    }
    #时间戳记录
    import time
    lt=time.localtime(time.time())
    #写入CSV文件
    filename="./validation/Noise_validation_n{}_iter{}_{}_{}_{}_{}.csv".format(n_components,iters,lt.tm_mon,lt.tm_mday,lt.tm_hour,lt.tm_min)
    csvFile=open(filename,'w+',newline="")
    csv_writer=csv.writer(csvFile)
    # 写入头
    csv_writer.writerow(["type","noise","params","origin SNR","SNR","PSNR","SSIM"])

    # 遍历验证
    for Noise in noises.keys():
        for params in noises[Noise]:
            print("[+] {} {}".format(Noise.__name__,params))
            if len(params)==1:
                ksvd_rst=denoise_ksvd(img,Noise(img,params[0]),n_components,iters,info="KSVD_Noise_{}_{}".format(Noise.__name__,params))
                reg_rst=denoise_reg_ksvd(img,Noise(img,params[0]),n_components,iters,info="regKSVD_Noise_{}_{}".format(Noise.__name__,params))
            elif len(params)==2:
                ksvd_rst=denoise_ksvd(img,Noise(img,params[0],params[1]),n_components,iters,info="KSVD_Noise_{}_{}".format(Noise.__name__,params))
                reg_rst=denoise_reg_ksvd(img,Noise(img,params[0],params[1]),n_components,iters,info="regKSVD_Noise_{}_{}".format(Noise.__name__,params))
            elif len(params)==3:
                ksvd_rst=denoise_ksvd(img,Noise(img,params[0],params[1],params[2]),n_components,iters,info="KSVD_Noise_{}_{}".format(Noise.__name__,params))
                reg_rst=denoise_reg_ksvd(img,Noise(img,params[0],params[1],params[2]),n_components,iters,info="regKSVD_Noise_{}_{}".format(Noise.__name__,params))
            # 写入csv
            csv_writer.writerow(["ksvd",Noise.__name__,params,ksvd_rst[0],ksvd_rst[1],ksvd_rst[2],ksvd_rst[3]])
            csv_writer.writerow(["reg ksvd",Noise.__name__,params,reg_rst[0],reg_rst[1],reg_rst[2],reg_rst[3]])

    csvFile.close()

def validation_multi_img(n_components,iters):
    ''' 
        针对多种原图 验证去噪效果
    '''
    imgs=["baboo.bmp","lenna.bmp","man.bmp","peppar.bmp"]

    import time
    lt=time.localtime(time.time())
    #写入CSV文件
    filename="./validation/img_validation_n{}_iter{}_{}_{}_{}_{}.csv".format(n_components,iters,lt.tm_mon,lt.tm_mday,lt.tm_hour,lt.tm_min)
    csvFile=open(filename,'w+',newline="")
    csv_writer=csv.writer(csvFile)
    # 写入头
    csv_writer.writerow(["type","img","params","origin SNR","SNR","PSNR","SSIM"])

    for img in imgs:
        o_img=cv2.imread('./images/original/{}'.format(img),0)
        n_img=SaltPepperNoise(o_img,0.1)
        ksvd_rst=denoise_ksvd(o_img,n_img,n_components,iters,info="KSVD_noise_saltPepper_0.1_img_{}".format(img))
        reg_rst=denoise_reg_ksvd(o_img,n_img,n_components,iters,info="regKSVD_noise_saltPepper_0.1_img_{}".format(img))
        csv_writer.writerow(["ksvd",img,ksvd_rst[0],ksvd_rst[1],ksvd_rst[2],ksvd_rst[3]])
        csv_writer.writerow(["reg ksvd",img,reg_rst[0],reg_rst[1],reg_rst[2],reg_rst[3]])
    csvFile.close()

if __name__ == "__main__":
    #validation_multi_noise(50,200)
    validation_multi_img(50,200)
