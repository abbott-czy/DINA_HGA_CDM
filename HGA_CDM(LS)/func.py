import csv
import math

import numpy as np
import time
import igraph
import pandas as pd

threshold_slip = 0.3
threshold_guess = 0.1
#tools 引用该库

# 读取试题*知识点考察矩阵
def acquireQ(Qpath):
    with open(Qpath, "r") as csvfile:
        reader = csv.reader(csvfile)
        Q = []
        temp = []  # 列表大小15（文件是15行）
        for line in reader:  # 按行读入 每一个line列表 大小11 知识点数是11
            temp.append(line)
        del temp[0]  # 删掉第一行序号行0、1、2...、10

        # csv中的数据原本是字符形式的，通过下面的两层循环将其转换为int类型方便之后的计算
        for line in temp:
            Qlie = []
            for i in line:
                i = int(float(i))
                Qlie.append(i)
            Q.append(Qlie)
    return Q


# 读取学生实际的答题情况
def acquireData(dataPath):

    with open(dataPath, "r") as csvfile:
        reader = csv.reader(csvfile)
        data = []
        temp = []
        for line in reader:
            temp.append(line)
        del temp[0]
        # 下面开始转化为数字型
        for line in temp:
            data_per_line = []
            for i in line:
                i = int(float(i))
                data_per_line.append(i)
            data.append(data_per_line)
    return data


# 从个体individual中提取学生*知识点掌握情况:
# 矩阵A为学生*知识点掌握情况，其中行为学生，列为知识点
def acquireA(individual, student, knowledge):
    A = individual[0:student * knowledge]  # GENE_LENGTH = student * knowlege + question * 2 * jingdu
    A = np.array(A)  # 转化为二维数组
    A = A.reshape(student, knowledge)
    # A = A.reshape(knowledge, student)  # 知识点数*学生数 换成：A.reshape(student, knowlege)?
    # A = A.transpose()  # 转置A矩阵：每个学生*知识点掌握情况 行为学生，列为知识点
    A = A.tolist()  # 二维列表
    return A


# 获取不考虑S和G的情况下，学生的答题情况矩阵:其行数为学生数，列数为题目数量
# A为学生*知识点、Q为题目*知识点
def acquireYITA(A, Q, student, question, knowledge): # A为I*K 能力属性掌握矩阵（随机初始化的） Q为 J*k 知识考察矩阵
    YITA = [[0] * question for i in range(student)]  # 产生二维全0的list:student*20  生成一个I*j列的学生作答0矩阵
    for i in range(student):
        for j in range(question):
            yita = 1
            for k in range(knowledge):
                yita = yita * pow(A[i][k], Q[j][k])  # a**Q  获得潜在作答矩阵
            YITA[i][j] = yita
    return YITA  # 二维列表：student*20 有用部分：student*question


# 获取存储所有s值的列表和存储所有g值的列表
def acquireSandG(student, individual, knowledge, GENE_LENGTH, len_s_g):
    S = []
    G = []
    num_points = pow(2, len_s_g)
    # 共question个i, GENE_LENGTH = student * knowledge + question * 2 * jingdu
    for i in range(knowledge * student, GENE_LENGTH, len_s_g * 2):  # stat，stop，step
        s = individual[i:i + len_s_g]
        s = list(map(str, s))  # map()会根据提供的函数对指定序列做映射 列表s转换为列表字符串
        s = "".join(s)  # 字符串连接
        sdec = int(s, 2)  # 二进制转换为十进制
        x = 0 + sdec * (threshold_slip / (num_points - 1))  # 失误率范围[0, 0.3]
        S.append(x)

        s = individual[i + len_s_g:i + (len_s_g * 2)]
        s = list(map(str, s))  # 转换为字符串
        s = "".join(s)  # 字符串连接
        sdec = int(s, 2)  # 二进制抓换为十进制
        x = 0 + sdec * (threshold_guess / (num_points - 1))  # 猜测率范围[0, 0.1]
        G.append(x)
    return S, G


# 获取考虑S，G的学生的答题情况矩阵X
def acquireX(student, question, YITA, S, G):
    X = [[0] * question for i in range(student)]  # 产生二维list:student*20
    Xscore = [[0] * question for i in range(student)]
    for i in range(student):
        for j in range(question):
            x = pow(G[j], (1 - YITA[i][j])) * pow((1 - S[j]), YITA[i][j])  # 引入s，g后掌握模式为η的概率
            Xscore[i][j] = x
            # X[i][j]=bernoulli.rvs(x)
            #TODO:参数会有影响
            if x >= 0.5341:
                X[i][j] = 1  # 答对
            else:
                X[i][j] = 0
    return X, Xscore   # X为引入s,g后的作答得分情况矩阵 , Xscore实际矩阵


# 汉明距离 码距 参数是种群
# 种群实质是个二维列表 里面每个列表之间两两计算码距
def hammingDis(invalid_ind):
    Matrix = []  # 二维列表
    for i in invalid_ind:
        sonMatrix = []
        for j in invalid_ind:
            dis = sum([ch1 != ch2 for ch1, ch2 in zip(i, j)])
            sonMatrix.append(dis)
        Matrix.append(sonMatrix)
    return Matrix

# 进行局部搜索的时候，判断可行性个体与记忆体中的个体距离大小
def distance(invalid_ind1,invalid_ind2):
    dis = sum([ch1 !=ch2 for ch1,ch2 in zip(invalid_ind1,invalid_ind2)])
    return dis
# 输入：待求种群、汉明距离、个体的基因长度
'''
是一个NBC算法，会对初始种群进行一个聚类的操作，
把一个初始种群聚类为几个小种群，再分别对每个小种群进行交叉变异等操作。
目的是能扩大搜索，找到更多相对最优解
'''


def getMultiPopList(invalid_ind, disMatrix, GENE_LENGTH):
    #invalid_ind = 100个列表，每个列表为其个体的染色体长度=600 , disMatrix 为每个个体到其他个体的距离100个列表
    # invalid_ind 首先第一轮传进来的无效价值的个体列表， disMatrix 汉明距离二维列表 ， GENE_LENGTH 染色体长度
    fitnessesList = [ind.fitness.values[0] for ind in invalid_ind]  # 各个个体的适应度
    indDict = dict(zip(range(len(invalid_ind)), fitnessesList))  # 字典：个体序号--适应度
    indDict = dict(sorted(indDict.items(), key=lambda x: x[1], reverse=True))  # 适应度排序
    # print("种群内序号-适应度字典",indDict)  # 排序后的字典{89: 292.0, 30: 286.0, 67: 286.0, 56: 284.0, ......
    sortInd = [i for i in indDict]  # 一维列表 适应度排序的序号
    # print("种群内排名",sortInd)  # 排序后的序号[89, 30, 67, 56, ......

    g = igraph.Graph(directed=True)
    g.add_vertices(sortInd)  # 添加这么多顶点
    g.vs['label'] = sortInd  # 顶点添加标签 以sortInd列表内容（即序号）为标签
    g.es['weight'] = 1.0  # 边长度

    # 算法3当中的的第一个for循环构建整个图 ###############
    index = 0
    weightEdgesDict = {}

    for i in sortInd[1:]: #得到的sortInd为适应度个体的排序，将从第i个开始，将排序大于他的个体，距离都设置为最大601 ，小于他的个体都形成边
        newsortInd = sortInd[index + 1:]  # 第一次执行时从下标1开始
        # print(newsortInd)
        idisListTemp = disMatrix[i]
        # j为不会比当前节点更好的节点，把他们对应的值置为最大
        for j in newsortInd:
            idisListTemp[j] = GENE_LENGTH + 1   #601
        # print(idisListTemp)
        # minDis返回的是最小的距离的下标，也就是要连接的节点编号
        minDisIndex = idisListTemp.index(min(idisListTemp)) #将刚得到的idisListTemp寻找他的最小值下边
        minDis = idisListTemp[minDisIndex]
        # print('最小距离的下标', minDisIndex)
        # print('最小距离', minDis)
        # 找到节点编号对应的下标
        # print('i', i)
        # print('minDisindex', minDisIndex)
        nodeIdSource = sortInd.index(i)
        nodeIdTarget = sortInd.index(minDisIndex)
        # nodeIdSource = i
        # nodeIdTarget = minDisIndex
        # print('nodeIdSource', nodeIdSource)
        # print('nodeIdTarget', nodeIdTarget)

        # 而g的add_edge方法建立边是根据下标建立的
        g.add_edge(nodeIdSource, nodeIdTarget)
        if (minDis == 0):
            g[nodeIdSource, nodeIdTarget] = 1
        else:
            g[nodeIdSource, nodeIdTarget] = minDis
        # # weightEdgesDIct是按节点下标存的
        # weightEdgesDict[(nodeIdSource, nodeIdTarget)] = minDis

        # weightEdgesDIct按真实编号存的
        weightEdgesDict[(i, minDisIndex)] = minDis
        # igraph.plot(g)
        index += 1
    # print(g.es['weight'])
    # 计算meanDIs
    meanDis = sum(g.es['weight']) / len(g.es['weight'])
    # print('meanDis', meanDis)
    # 计算度
    numbers = g.indegree()
    # print(numbers)
    # neighbors存了每个节点的度，是按真实节点的名称存的
    neighbors = dict(zip(g.vs['label'], numbers))

    # print("weightEdgesDict",weightEdgesDict)
    # # print(len(weightEdgesDict))
    # print("neighbors",neighbors)
    # igraph.plot(g)
    # 删除边：删除时是按下标去删除的
    for node in weightEdgesDict:
        # print(node)
        # print(weightEdgesDict[node])
        # ni = neighbors[sortInd[node[0]]]
        ni = neighbors[node[0]]
        # print('ni',ni)
        nodeIdSource = sortInd.index(node[0])
        nodeIdTarget = sortInd.index(node[1])
        if (weightEdgesDict[node] > meanDis and ni > 2):  # and ni > 10
            # print(node)
            g.delete_edges((nodeIdSource, nodeIdTarget))

    # igraph.plot(g).save('hahaah.png')
    newg = g.as_undirected().decompose()
    return newg


def computeTime():
    now = time.time()
    local_time = time.localtime(now)
    date_format_localtime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    return date_format_localtime


def decode_slip(x, len_s_g=7):  # 失误率解码函数
    # x为失误率
    min_x = 0.0
    num_points = pow(2, len_s_g)
    n = (x - min_x) * (num_points - 1) / threshold_slip
    if(math.isnan(n)):
        print('here a nan')

    num_int = int(n)    #在这里出现nan异常h
    tmp_str = '{:0' + str(len_s_g) + 'b}'
    numbers = tmp_str.format(num_int)
    n = list(map(int, numbers))
    return n


def decode_guess(x, len_s_g=7):  # 猜测率解码函数
    min_x = 0.0
    num_points = pow(2, len_s_g)
    n = (x - min_x) * (num_points - 1) / threshold_guess
    num_int = int(n)
    tmp_str = '{:0' + str(len_s_g) + 'b}'
    numbers = tmp_str.format(num_int)
    n = list(map(int, numbers))
    return n


def txt2csv(file):
    txt = np.loadtxt(file+".txt")
    print(txt.shape)
    txtDF = pd.DataFrame(txt)
    txtDF.to_csv(file+".csv", index=False)
