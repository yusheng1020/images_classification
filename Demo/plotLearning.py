import matplotlib.pyplot as plt

# test_score 测试集的精度
y1_i = [32.01377033999997, 52.995126360000086,185.63600004000017]
y1_c = [34.805954499999984,59.619636000000014,188.34518989999992]
y1_a = [34.05963100000008, 53.854412799999864,187.09286480000037]
# macro-F1 测试集的宏F1
y2_i = []
y2_c = []
y2_a = []
# micro-F1 测试集的微F1
y3_i = []
y3_c = []
y3_a = []
# # 隐藏层神经元的个数
# x1 = []
# 隐藏层的个数
x2 = [100,200,400]
# # 窗口大小
# x3 = []

ax1 = plt.gca()
ax1.set(xlabel='neuron_num', ylabel='train_seconds')
# ax1.set(xlabel='batch_size', ylabel='accuracy')
# ax1.set(xlabel='batch_size', ylabel='macro-F1')
# ax1.set(xlabel='batch_size', ylabel='micro-F1')

# 三种不同学习率
l1, = ax1.plot(x2,y1_i,'red', label='invscaling')
l2, = ax1.plot(x2,y1_c,'green', label='constant')
l3, = ax1.plot(x2,y1_a,'blue', label='adaptive')
plt.legend()
plt.show()