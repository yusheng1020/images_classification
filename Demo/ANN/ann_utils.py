from sklearn.neural_network import MLPClassifier
from KNN.knn_utils import printReport
import time

def ann_train(x_test,y_test,x_train, y_train,units,max_iter,batch_size,learning_rate):# 可以继续加参数 具体可参考 http://www.scikitlearn.com.cn/0.21.3/18/#1172 与源码部分
    """
    :param x_test: 测试集
    :param y_test: 测试集标签
    :param x_train: 训练集
    :param y_train: 训练集标签
    :param units: 神经元层数分布（[x,y,...,z]个数为层数 数值为层内神经元数）
    :param max_iter: 最大迭代数
    :param batch_size: 窗口
    :param learning_rate:学习率
    :return:
    """
    # 激活函数：relu, logistic, tanh
    # 优化算法：lbfgs, sgd, adam。adam适用于较大的数据集，lbfgs适用于较小的数据集。
    # 初始化模型
    #MLPClassifier多层感知器分类器
    # hidden_layer_size: 元组(20,40,)第i个元素表示的是第i层隐藏层的神经元的个数
    # solver: 优化器，三种优化算法
    # activation: 激活函数，三种激活函数
    print(units)
    ann_model = MLPClassifier(hidden_layer_sizes=units, activation='logistic', solver='adam', random_state=0,batch_size=batch_size,max_iter=max_iter,learning_rate=learning_rate)
    # 训练模型
    ann_model.fit(x_train, y_train)
    report_str, test_score, macro, micro = printReport(x_test, y_test, x_train, y_train, ann_model)
    return report_str,test_score,macro,micro

def ann_able_method(X_test,Y_test,X_train, Y_train,flieName,number,learning_rate,units,max_iter,batch_size):# 根据官方文档加参数
    """
    :param X_test: 测试集
    :param Y_test: 测试集标签
    :param X_train: 训练集
    :param Y_train: 训练集标签
    :param num_classes: 类数
    :param flieName: 写文件名
    :param number: 训练次数
    """
    f = open(flieName, 'a+')
    print("layers:" + str(units))
    # 写入隐藏层的信息
    f.write("layers:" + str(units))
    f.write("\n")
    # 写入重复的次数
    print("number:" + str(number))
    f.write("number:" + str(number))
    f.write("\n")
    # 写入迭代次数
    print("iter_num:" + str(max_iter))
    f.write("iter_num:" + str(max_iter))
    f.write("\n")
    # 写入学习率
    print("learning_rate:" + str(learning_rate))
    f.write("learning_rate:" + str(learning_rate))
    f.write("\n")
    # 写入窗口大小
    print("batch_size:" + str(batch_size))
    f.write("batch_size:" + str(batch_size))
    f.write("\n")
    sum = 0
    ma = 0
    mi = 0
    timeSum = 0
    for j in range(0,number):
        start = time.perf_counter()
        report_str,test_score,macro,micro = ann_train(X_test,Y_test,X_train, Y_train,units,max_iter,batch_size,learning_rate)
        stop = time.perf_counter()
        sum = sum + test_score
        ma = ma + macro
        mi = mi + micro
        timeSum = timeSum + stop - start
        print("time = " + str(stop - start) + "s")
        f.write(report_str + "time = " + str(stop - start) + "s")
        f.write("\n")
    print("average score = " + str(sum/number) + " average Macro f1 = " + str(ma/number) + " average Micro f1 = " + str(mi/number)  + "s")
    f.write("average score = " + str(sum/number) + " average Macro f1 = " + str(ma/number) + " average Micro f1 = " + str(mi/number) + "average time = " + str(timeSum/number) + "s")
    f.write("\n")
    f.write("----------------------------------------------------------------------------------------------------------------")
    f.write("\n")
    f.close()