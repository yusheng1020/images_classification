## 代码结构说明

### ANN：
#### ann_utils.py
- 通过规定隐藏层个数，隐藏层内神经元节点个数，窗口大小，迭代次数，学习率，使用`ann_train`函数调用`MLPClassifier`分类器函数来构建我们的人工神经网络模型，并计算相关模型的评估参数。
- 在`ann_able_method`中我们对模型的效果多次计算，取平均值，并将相关数据写入文件中记录。
  
### KNN：
#### knn_utils.py
- 通过规定邻居节点个数k，使用`knn_train`函数调用`KNeighborsClassifier`分类器函数来实现我们的`KNN`模型，并计算相关模型的评估参数
- 在`knn_able_method`中我们对模型的效果多次计算，其评估参数结果取均值，并将相关数据写入文件中记录。
  
### data
#### test
- 测试集数据
#### train 
- 训练集数据

### reading_data.py
- 按照对应结构对测试集使用`reading_test_data`方法对第index个文件进行数据读入。
- 按照对应结构对训练集使用`reading_train_data`方法对第index个文件进行数据读入。

### main.py
- 测试代码，合并数据与建立模型并训练得出指标
  
### plotLearning.py
- 绘图代码，将指标数据绘制得到曲线对比图

## 运行说明

- 运行main.py进行ANN与KNN的模型建立与测试得出指标
- 运行plotLearning.py进行对指标值的对比图绘制
