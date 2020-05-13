#============================================================
# 对去噪效果进行多方面测试
#
#===========================================================
import time
import csv
from DCTlearning import KSVD
from ImageTools import *

# 打开图片
img=cv2.imread('.\images\lenna.bmp',0)
# 生成图片噪声 默认为高斯噪声
n_img=GaussianNoise(img,1.0,20)

def DCT_params_test():
    ''' 对模型的各种参数进行测试 '''
    # 打开csv文件
    lt=time.localtime(time.time())
    filename="./validation/DCT_params_test_gauss_{}_{}_{}_{}.csv".format(lt.tm_mon,lt.tm_mday,lt.tm_hour,lt.tm_min)
    csvFile=open(filename,'w+',newline="")
    csv_writer=csv.writer(csvFile)
    # 写入头
    csv_writer.writerow(["components","iter","tol","SNR","PSNR","SSIM"])

    # 遍历参数范围
    for ncomponent in range(20,150,10):
        # 字典原子个数
        for _iter in range(20,100,10):
            # 训练更新次数
            for tol in [1e-5,1e-6,1e-7]:
                print("[*] {} {} {} ....".format(ncomponent,_iter,tol))
                ksvd=KSVD(n_img,n_components=ncomponent,max_iter=_iter,tol=tol)
                ksvd.fit()
                de_img=ksvd.denoise(show_cmp=False,show_cmp_save_dir=None)

                # 写入csv
                csv_writer.writerow([ncomponent,_iter,str(tol),cal_SNR(de_img),cal_PSNR(img,de_img),cal_SSIM(img,de_img)])
    
    csvFile.close()

def DCT_muti_noise_test(n_components=80,max_iter=100,tol=1e-5):
    ''' 对各种噪声类型进行去噪效果测试 '''
    # 打开csv文件
    lt=time.localtime(time.time())
    filename="./validation/DCT_noise_test_{}_{}_{}_{}.csv".format(lt.tm_mon,lt.tm_mday,lt.tm_hour,lt.tm_min)
    csvFile=open(filename,'w+',newline="")
    csv_writer=csv.writer(csvFile)

    # 写入头
    csv_writer.writerow(["noise type","noise params","noise SNR","SNR","PSNR","SSIM"])

    # 各种噪声类型以及对应的测试参数
    Noise={
        # GaussianNoise:[ (1.0,5),(1.0,10),(1.0,20),(1.0,30),(1.0,40),
        #                 (0.8,5),(0.8,10),(0.8,20),(0.8,30),(0.8,40),
        #                 (0.5,5),(0.5,10),(0.5,20),(0.5,30),(0.5,40),
        #                 (0.2,5),(0.2,10),(0.2,20),(0.2,30),(0.2,40),
        #                 (0.1,5),(0.1,10),(0.1,20),(0.1,30),(0.1,40),],
        # SaltPepperNoise:[1.0,0.9,0.7,0.5,0.2,0.1],
        # RayleighNoise:[ (5,1.0),(10,1.0),(20,1.0),(50,1.0),(100,1.0),
        #                 (5,0.8),(10,0.8),(20,0.8),(50,0.8),(100,0.8),
        #                 (5,0.5),(10,0.5),(20,0.5),(50,0.5),(100,0.5),
        #                 (5,0.2),(10,0.2),(20,0.2),(50,0.2),(100,0.2),
        #                 (5,0.1),(10,0.1),(20,0.1),(50,0.1),(100,0.1)],
        # ExponentialNoise:[  (5,1.0),(10,1.0),(20,1.0),(50,1.0),(100,1.0),(200,1.0),
        #                     (5,0.8),(10,0.8),(20,0.8),(50,0.8),(100,0.8),(200,0.8),
        #                     (5,0.5),(10,0.5),(20,0.5),(50,0.5),(100,0.5),(200,0.5),
        #                     (5,0.2),(10,0.2),(20,0.2),(50,0.2),(100,0.2),(200,0.2),
        #                     (5,0.1),(10,0.1),(20,0.1),(50,0.1),(100,0.1),(200,0.1),],
        # UniformNoise:[
        #     (0,10,0.8),(0,20,0.8),(0,50,0.8),(0,100,0.8),
        #     (0,10,0.5),(0,20,0.5),(0,50,0.5),(0,100,0.5),
        #     (0,10,0.2),(0,20,0.2),(0,50,0.2),(0,100,0.2),
        #     (0,10,0.1),(0,20,0.1),(0,50,0.1),(0,100,0.1),]
        GammaNoise:[(1,0.5,0.8),(1,1.0,0.8),(1,2.0,0.8),(1,10.0,0.8),
                        (2,0.5,0.8),(2,1.0,0.8),(2,2.0,0.8),(2,10.0,0.8),
                        (5,0.5,0.8),(5,1.0,0.8),(5,2.0,0.8),(5,10.0,0.8),
                        (9,0.5,0.8),(9,1.0,0.8),(9,2.0,0.8),(9,10.0,0.8),
                    (1,0.5,0.5),(1,1.0,0.5),(1,2.0,0.5),(1,10.0,0.5),
                        (2,0.5,0.5),(2,1.0,0.5),(2,2.0,0.5),(2,10.0,0.5),
                        (5,0.5,0.5),(5,1.0,0.5),(5,2.0,0.5),(5,10.0,0.5),
                        (9,0.5,0.5),(9,1.0,0.5),(9,2.0,0.5),(9,10.0,0.5),
                    (1,0.5,0.3),(1,1.0,0.3),(1,2.0,0.3),(1,10.0,0.3),
                        (2,0.5,0.3),(2,1.0,0.3),(2,2.0,0.3),(2,10.0,0.3),
                        (5,0.5,0.3),(5,1.0,0.3),(5,2.0,0.3),(5,10.0,0.3),
                        (9,0.5,0.3),(9,1.0,0.3),(9,2.0,0.3),(9,10.0,0.3),
                    (1,0.5,0.1),(1,1.0,0.1),(1,2.0,0.1),(1,10.0,0.1),
                        (2,0.5,0.1),(2,1.0,0.1),(2,2.0,0.1),(2,10.0,0.1),
                        (5,0.5,0.1),(5,1.0,0.1),(5,2.0,0.1),(5,10.0,0.1),
                        (9,0.5,0.1),(9,1.0,0.1),(9,2.0,0.1),(9,10.0,0.1),
                        ]
    }

    # 使用最佳的参数训练字典
    for noise in Noise.keys():
        for params in Noise[noise]:
            print("[+] noise:{} params:{}".format(noise.__name__,params))
            #生成噪声图像
            if isinstance(params,float):
                n_img=noise(img,params)
            elif len(params)==2:
                n_img=noise(img,params[0],params[1])
            elif len(params)==3:
                n_img=noise(img,params[0],params[1],params[2])
            # 训练字典并去噪
            ksvd=KSVD(n_img,n_components,max_iter,tol)
            ksvd.fit()
            de_img=ksvd.denoise(show_cmp=False,show_cmp_save_dir=None)

            # 写入csv
            csv_writer.writerow([noise.__name__,params,cal_SNR(n_img),cal_SNR(de_img),cal_PSNR(img,de_img),cal_SSIM(img,de_img)])
    csvFile.close()
if __name__ == "__main__":
    #DCT_params_test()
    DCT_muti_noise_test()



