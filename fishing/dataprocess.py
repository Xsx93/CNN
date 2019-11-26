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
def lineprocess(line,sample_index,indexdata):
    if len(line)!=len(indexdata):
        print("error! lenth of line not match indexdata")
    for i in range(len(line)):
        for j in range(indexdata[i]):
            index0 = indexdata[i][j]
            indexdata[i][j] = line[indexdata[i][j]]
    return indexdata
print(indexmat(30))

# for h in range(30):
#     for g in range(30):
#         print(str(indexda(30)[h][g]).rjust(5), end='')
#     print('')