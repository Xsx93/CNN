#%%
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from numpy import *
from tqdm import tqdm
import scipy.io as scio
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

fea_num = 30
cla_num = 2
# fea_row = 5
# fea_col = 6
# fea_row1 = 3
# fea_col1 = 3
# fea_row2 = 2
# fea_col2 = 2

fea_row = 1
fea_col = 30
fea_row1 = 7
fea_col1 = 7
fea_row2 = 1
fea_col2 = 8




N = 1
totalline = 11055//N
trainline = 8000//N
testline = 11000//N
steplenth = 1536
batchnum = 5000
kp1 = 1
kp = 0.75

learningrate = 4e-4
# con = 1
deep1 = 10#32
deep2 = 20#64
outdeep = 64#1024



# fr = open('./alldata.txt')  #打开数据文件
#
# dataMat = zeros((totalline, 30), dtype=float)  # 先创建一个 3x3的全零方阵A，并且数据的类型设置为float浮点型
# labelMat = zeros((totalline, 1), dtype=float)
#
# lines = fr.readlines()  # 把全部数据文件读到一个列表lines中
# A_row = 0  # 表示矩阵的行，从0行开始
# for line in lines:  # 把lines中的数据逐行读取出来
#     if A_row>totalline:
#         continue
#     line = line.replace(',', " ")
#     list = line.strip().split()  # 处理逐行数据：strip表示把头尾的'\n'去掉，split表示以空格来分割行数据，然后把处理后的行数据返回到list列表中
#     dataMat[A_row,:] = list[0:30]  # 把处理后的数据放到方阵A中。list[0:3]表示列表的0,1,2列数据放到矩阵A中的A_row行
#     labelMat[A_row,:] = list[30:31]
#     A_row += 1  # 然后方阵A的下一行接着读
#     print(A_row)
#     # print(labelMat)
#
# print(type(labelMat))
# print("ok1")




dataFile1 = 'Fishing3.mat'  # mat文件路径，data为名字，其中的变量名为A
data = scio.loadmat(dataFile1)
dataMat = data['feature']
labelMat = data['label']
# print(labelMat)
# labelMat += 1
# labelMat /= 2
# print(labelMat)


trainset = dataMat[:trainline, :]
trainlabel = labelMat[:trainline, :]
testset = dataMat[trainline:testline, :]
testlabel = labelMat[trainline:testline, :]
print("ok2")


# train_csv_name = './daima/dt.xlsx'
# df = pd.read_excel(train_csv_name)
# df = DataFrame(df)
# print(df)
# train_data = df.iloc[:8000, 0: 30]
# print(train_data)
# train_label = df.iloc[:8000, -1]
# test_data = df.iloc[8000:, 0: 30]
# test_label = df.iloc[8000:, -1]
# data_train_DF.shape
# 输出为numpy array类型
# trainset = np.array(train_data)
#
# trainlabel = np.array(train_label)
# testset = np.array(test_data)
# testlabel = np.array(test_label)



def next_batch(datset,i,nextnum,total):
    # now = i % (trainline // nextnum)
    # # if (now+nextnum) > total:
    # #     return datset[:nextnum, :]
    # # else:
    # print(datset)
    # return datset[now:(now+nextnum), :]
    now = i % (total // nextnum - 1)
    return datset[(now * nextnum):(now * nextnum + nextnum), :]

sess = tf.InteractiveSession()


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
x = tf.placeholder(tf.float32, [None, fea_num])  #
y_ = tf.placeholder(tf.float32, [None, cla_num])
x_image = tf.reshape(x, [-1,fea_row ,fea_col ,1 ])
                        
W_conv1 = weight_variable([fea_row1 , fea_col1 , 1 , deep1 ])
b_conv1 = bias_variable([deep1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)
print(h_pool1)


W_conv2 = weight_variable([fea_row1 , fea_col1 , deep1 , deep2 ])
b_conv2 = bias_variable([deep2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2)

W_fc1 = weight_variable([fea_row2 * fea_col2 * deep2 , outdeep ])
b_fc1 = bias_variable([outdeep])
h_pool2_flat = tf.reshape(h_pool2, [-1, fea_row2 * fea_col2 * deep2 ])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1 , keep_prob)

W_fc2 = weight_variable([outdeep , cla_num ])
b_fc2 = bias_variable([cla_num ])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learningrate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()
for i in range(batchnum):
    # j = i % (trainline)
    batch0 = next_batch(trainset, i, steplenth, trainline)

    batch1 = next_batch(trainlabel, i, steplenth, trainline)
    if i % 4 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch0, y_: batch1, keep_prob:kp1})
    # print('x', x)
    # print('y_', y_)
    # print(type(batch0))
    # print(type(batch1))
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch0, y_: batch1, keep_prob: kp})
test_accuracy = accuracy.eval(feed_dict = {x: testset, y_: testlabel, keep_prob: kp1})
print('the test accuracy :{}'.format(test_accuracy))
saver = tf.train.Saver()
path = saver.save(sess, './my_net/fishdect_deep.ckpt')
print('save path: {}'.format(path))



