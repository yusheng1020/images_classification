from reading_data import reading_train_data,reading_test_data
import numpy as np
from ANN.ann_utils import ann_train,ann_able_method
from KNN.knn_utils import knn_train,knn_able_method

if __name__ == '__main__':
        x_test,y_test = reading_test_data(0)
        x_train,y_train = reading_train_data(0)
        for i in range(1, 200):
            x_test_pro, y_test_pro = reading_test_data(i)
            x_train_pro, y_train_pro = reading_train_data(i)
            x_test = np.concatenate((x_test,x_test_pro))
            y_test = np.concatenate((y_test, y_test_pro))
            x_train = np.concatenate((x_train, x_train_pro))
            y_train = np.concatenate((y_train, y_train_pro))
        y_test = y_test.tolist()
        y_train = y_train.tolist()
        ann_fileName = "ANN.txt"
        knn_fileName = "KNN.txt"
        # 单层神经元节点个数
        a = (100,)
        b = (200,)
        c = (400,)
        # 多层隐藏层
        # a = (50,)
        # b = (50,50,)
        # c = (50,50,50,)
        units_list = [a,b,c]  #存储三种隐藏层的信息
        learning_rate_list = ['invscaling', 'constant', 'adaptive'] #存储三种学习率
        max_iter_list = [500,5000,10000 ] #存储三种迭代次数
        batch_size_list = [50,100,150] #存储三种窗口大小
        k_nums = [1,2,3,4,5,6,7,8,9,10] #KNN算法的k个邻居的数量
        # train_nums = [5,6,7,8,9,10] #KNN训练的次数
        for i in range(0,3):
            for j in range(0,3):
                for k in range(0,3):
                    for l in range(0,3):
                        ann_able_method(x_test,y_test,x_train,y_train,ann_fileName,1,learning_rate_list[i],units_list[j],max_iter_list[k],batch_size_list[l])
        # ann_able_method(x_test,y_test,x_train,y_train,"666.txt",5,'constant',(50,),100,200)
        for i in range(0,10):
                knn_able_method(x_test, y_test, x_train, y_train, k_nums[i], knn_fileName, 5)

