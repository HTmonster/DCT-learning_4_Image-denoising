#===============字典学习的几个相关概念===========================
# 1.[原始样本]  "以前的知识 Y"
# 2.[字典矩阵]  "字典D"   [原子]  "字典中的词条 列向量dk"
# 3.[稀疏矩阵]  "查字典的方法 X"
# 4.[矩阵乘法]  "查字典的过程 DX"
# 
# 主要思想：利用字典矩阵 稀疏线性表示原始样本
# 
#===============================================================
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import copy
import math
from sklearn import linear_model

from  ImageTools import *


class KSVD:
    ''' KSVD 字典学习 '''
    def __init__(self,pic,n_components,max_iter=30, tol=1e-6,
                 n_nonzero_coefs=None):
        '''
        :param pic : 用来训练的原图像
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        '''

        self.DCT=None        #字典
        self.sparsecode=None #稀疏矩阵
        
        self.pic=pic
        self.n_components=n_components
        self.max_iter=max_iter 
        self.tol = tol
        self.n_nonzero_coefs = n_nonzero_coefs 

    def OMP(self,X,y,n_nonzero_coefs=None):
        ''' 稀疏编码 

        OMP算法(正交匹配追踪) 
        =============================================
        这里借助 sklearn 模块实现
        =============================================
        '''
        return linear_model.orthogonal_mp(X,y,n_nonzero_coefs)

    def creat_DCT(self):
        ''' 初始化字典 
        =============================================
        从原始样本Y∈Rm×n中
            1.随机取K个列向量
        --> 2.(或) 左奇异矩阵的前K的列向量{d1,d2...dk}
        作为初始字典的原子D(0)∈Rm×K
        =============================================
        '''
        u,s,v = np.linalg.svd(self.pic)
        self.DCT=u[:,:self.n_components]

    def _update_DCT(self,y,d,x):
        ''' 字典更新 
        KSVD 算法
        ============================================
        逐列更新字典 D(j) 字典的列dn∈{d1,d2,d3....dk}

        1.计算误差矩阵Ek
        2.取出稀疏矩阵的第k个行向量x^k_T不为0的索引的集合wk 取出Ek对应不为0的列 得到E’k
        3.对E'K做奇异值分解 Ek=UΣVT  
            3.1 取U的第1列更新字典的第k列
            3.2 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
        =============================================
        '''

        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            # 更新第i列
            d[:,i] = 0
            # 计算误差矩阵
            r = (y-np.dot(d,x))[:,index]
            # 利用svd 方法 求解更新字典和稀疏系数矩阵
            u,s,v = np.linalg.svd(r,full_matrices=False)
            # 更新字典
            d[:,i]=u[:,0]
            # 更新稀疏系数矩阵
            for j,k in enumerate(index):
                x[i,k] = s[0] * v[0,j]
        return d,x
    
    def fit(self):
        ''' 迭代构建字典
        ==============================================
        0. 根据输入图片初始化字典
        1. 根据迭代次数更新字典
            1.1 计算稀疏矩阵X
            1.2 计算表示误差
                1.2.1 若是误差满足预期 则停止
                1.2.2 否则更新字典D
            1.3 计算稀疏矩阵X
        ==============================================
        '''
        #0.初始化字典
        self.creat_DCT()
        print("[+]==============created DCT================")

        #1.迭代更新字典
        for i in range(self.max_iter):
            print("[*] update DCT {}/{}".format(i,self.max_iter))
            # 1.1 稀疏编码x
            x=self.OMP(self.DCT,self.pic,self.n_nonzero_coefs)
            # 1.2 计算误差
            e = np.linalg.norm(self.pic - np.dot(self.DCT, x))
            # 1.2.1 误差满足条件  停止
            if e < self.tol:
                break
            # 1.2.2 否则更新字典
            self._update_DCT(self.pic,self.DCT,x)
        
        # 2.计算稀疏矩阵
        self.sparsecode=self.OMP(self.DCT,self.pic,self.n_nonzero_coefs)

        print("[+]===========DCT fited===================")

        return self.DCT,self.sparsecode
    
    def SparseReduction(self,show_cmp=True,show_cmp_save_dir='./images/denoise/'):
        ''' 图像稀疏还原
        :params show_cmp: boolean 是否显示对比图画
        :return 还原后的图像
        ====================================
        查字典 DX 字典D*稀疏矩阵X
        '''

        # 稀疏还原
        imgReduction=np.array(np.dot(self.DCT,self.sparsecode),dtype=int)

        if show_cmp:
            # 显示对比图像
            plt.figure()
            
            plt.subplot(1,2,1)
            plt.imshow(self.pic,cmap ='gray')
            plt.text(-100,-100,"n_components:{}".format(self.n_components))
            plt.text(-100,-60,"max_iter:{}".format(self.max_iter))
            plt.text(-100,-20,"tol:{}".format(self.tol))
            plt.title("noise")
            plt.subplot(1,2,2)
            plt.imshow(imgReduction,cmap ='gray')
            plt.title("denoise")
            if show_cmp_save_dir:
                plt.savefig("{}\KSVD_n{}_iter{}_tol{}.png".format(
                    show_cmp_save_dir,self.n_components,self.max_iter,self.tol))
            plt.show()
    def denoise(self,show_cmp=True,show_cmp_save_dir='.\images\denoise'):
        ''' 图像去噪 
        其实也就是字典稀疏还原
        '''
        self.SparseReduction(show_cmp,show_cmp_save_dir)

if __name__ == "__main__":
    # 原正常图像
    img=cv2.imread('.\images\lenna.bmp',0)
    # 生成噪声图像
    gaussImg=GaussianNoise(img,1.0,20)

    # KSVD字典学习类
    ksvd=KSVD(gaussImg,64)
    # 学习 更新字典
    ksvd.fit()
    # 稀疏还原 也就是去噪
    ksvd.denoise()



