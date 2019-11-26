import numpy as np
import random
import scipy.io as scio

def stepnum(n):
    res = 0
    for k in range(n+1):
        res+=k
    return res

def indexmat(featurenum):
    indexdata = [[] for l in range(featurenum)]
    for i in range(featurenum):
        for j in range(featurenum):
            indexdata[i].append((stepnum(i)+j)%featurenum)
    return indexdata
def lineprocess(line,indexdata,datamat):
    if len(line)!=len(indexdata):
        print("error! lenth of line not match indexdata",len(line),len(indexdata))
    for i in range(len(indexdata)):
        for j in range(len(indexdata[i])):
            datamat[i][j] = line[indexdata[i][j]]
    # print(datamat)
    return indexdata


# for h in range(30):
#     for g in range(30):
#         print(str(indexda(30)[h][g]).rjust(5), end='')
#     print('')

if __name__ == '__main__':
    with open('./alldata.txt','r') as f:
        lines = f.readlines()
    fea_num = 30
    cla_num = 2
    trainline = 8000
    testline = 11000
    totalline = len(lines)
    dataMat = np.zeros((totalline, fea_num ,fea_num), dtype=int)  # 先创建一个 totalline * fea_num * fea_num的全零方阵A，并且数据的类型设置为float浮点型
    labelMat = np.zeros((totalline, 2), dtype=int)
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    A_row = 0  # 表示矩阵的行，从0行开始
    indexdata = indexmat(fea_num)  # 制作映射索引矩阵
    print(indexdata)
    for line in lines:  # 把lines中的数据逐行读取出来
        if A_row > totalline:
            continue
        line = line.replace(',', " ")
        line = line.strip().split()
        line = [int(x) for x in line ]
        # print(line)
        list = line[0:30]
        print("=====================",A_row)
        lineprocess(list,indexdata,dataMat[A_row])


        print(type(line[30]))
        if line[30] == -1:
            labelMat[A_row][0] = 1
            # print("-1")
        elif line[30] == 1:
            labelMat[A_row][1] = 1
            # print("1")

        A_row+=1

    trainset = dataMat[:trainline, :,:]
    trainlabel = labelMat[:trainline, :]
    testset = dataMat[trainline:testline,:, :]
    testlabel = labelMat[trainline:testline, :]
    print("ok")
    print(dataMat)
    print(labelMat)

    Embedding_UI = 'Fishing_30x30.mat'
    # N是需要保存的矩阵，A为字典名，读取的方便，保存为新矩阵dataNew=Embedding_UI
    scio.savemat(Embedding_UI, {'feature': dataMat, 'label': labelMat})

    # print(dataMat)