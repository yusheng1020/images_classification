from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import time

def knn_train(X_test,Y_test,X_train, Y_train,num_classes):# 可以继续加参数 具体可参考 http://www.scikitlearn.com.cn/0.21.3/7/#1661 与源码部分
    """
    :param X_test: 测试集
    :param Y_test: 测试集标签
    :param X_train: 训练集
    :param Y_train: 训练集标签
    :param num_classes: 类数
    :return: 报告文本,验证score,macro,micro
    """
    model = KNeighborsClassifier(n_neighbors=num_classes)# 添加参数，参照官方文档调参
    model.fit(X_train, Y_train)
    report_str,test_score,macro,micro = printReport(X_test, Y_test, X_train, Y_train, model)
    return report_str,test_score,macro,micro

def printReport(X_test,Y_test,X_train, Y_train,model):
    """
    :param X_test: 测试集
    :param Y_test: 测试集标签
    :param X_train: 训练集
    :param Y_train: 训练集标签
    :param model: 模型
    :return: 报告文本,验证score,macro,micro
    """
    # 用训练数据评判模型
    train_score = model.score(X_train, Y_train)
    # 用测试数据去评判模型
    test_score = model.score(X_test, Y_test)
    #取值范围均为[0,1]
    #宏F1，所有混淆矩阵的P和R都算出来，求得的平均P和R计算出的F1
    macro = f1_score(Y_test, model.predict(X_test), average='macro')
    #微F1,所有混淆矩阵内的数都算出来，求得的平均混淆矩阵计算出来的P和R计算出来的F1
    micro = f1_score(Y_test, model.predict(X_test), average='micro')
    print('train score: {}'.format(train_score) + '|' + "Train Macro f1:" + str(macro)+ "|" + "Train Micro f1:" +str(micro) + "|" + "score = " + str(test_score))
    return 'train score: {}'.format(train_score) + '|' + "Train Macro f1:" + str(macro)+ "|" + "Train Micro f1:" +str(micro) + "|" + "score = " + str(test_score) , test_score,macro,micro

def knn_able_method(X_test,Y_test,X_train, Y_train,num_classes,flieName,number):# 根据官方文档加参数
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
    sum = 0
    ma = 0
    mi = 0
    timeSum = 0
    # 写入k的个数
    print("k_num:" + str(num_classes))
    f.write("k_num:" + str(num_classes))
    f.write("\n")
    # 写入训练次数
    print("number:" + str(number))
    f.write("number:" + str(number))
    f.write("\n")
    for j in range(0,number):
        start = time.perf_counter()
        report_str,test_score,macro,micro = knn_train(X_test,Y_test,X_train, Y_train,num_classes)
        stop = time.perf_counter()
        sum = sum + test_score
        ma = ma + macro
        mi = mi + micro
        timeSum = timeSum + stop - start
        print("time = " + str(stop - start) + "s")
        f.write(report_str + "time = " + str(stop - start) + "s")
        f.write("\n")
    print("average score = " + str(sum/number) + " average Macro f1 = " + str(ma/number) + " average Micro f1 = " + str(mi/number) + "average time = " + str(timeSum/number) + "s")
    f.write("average score = " + str(sum/number) + " average Macro f1 = " + str(ma/number) + " average Micro f1 = " + str(mi/number) + "average time = " + str(timeSum/number) + "s")
    f.write("\n")
    f.write("----------------------------------------------------------------------------------------------------------------")
    f.write("\n")
    f.close()