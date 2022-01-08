import numpy as np
import pandas as pd
import time
import math
from multiprocessing import Pool
from sklearn.model_selection import KFold
import csv
import random


'''
use math2015 data,including FrcSub,Math1,Math2
training data use 80% of total data
'''

threshold_slip = 0.15
threshold_guess = 0.05

def Write_data(data,i):
    di, dj = data.shape
    filename = "czy_results/ceshi/"+ "_"  + "_th_" + str(i) + ".txt"
    f_record = open(filename, 'a')
    for i in range(di):
        f_record.write("id"+str(i)+":  ")
        for j in range(dj):
            f_record.write(str(data[i][j]) + '  ')
        f_record.write('\n')
    f_record.close()

def EStep(IL,sg,n,r,k,i):
    base = 2**(k-2)
    for l in range(i*base,(i+1)*base):
        # student number
        lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
            1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    return IL

def MStep(IL,n,r,k,i):
    base = 2**(k-2)
    ni,nj=n.shape
    IR = np.zeros((4, nj))
    n1 = np.ones(n.shape)
    for l in range(i*base,(i+1)*base):
        IR[0] += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
        IR[1] += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
        IR[2] += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
        IR[3] += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
    return IR
def trainDINAModel(n,Q,threshold):
    startTime = time.time()
    #print('*************开始 DINA模型 训练*************')
    ni, nj = n.shape   #取学生得分矩阵N和知识点考察矩阵Q的行与列 ni=482 nj=20
    Qi, Qj = Q.shape   #Qi:20 Qj:8

    #构造K矩阵，K矩阵表示k个技能可以组成的技能模式矩阵 ，Qj个技能，就可能有2^Qj的技能掌握情况
    K = np.mat(np.zeros((Qj, 2 ** Qj), dtype=int))   #dtype返回元素的数据类型改为int型8*256矩阵全零 k的一列代表一种知识点掌握装填
    for j in range(2 ** Qj):
        l = list(bin(j).replace('0b', ''))  #L代表每一种训练模式
        for i in range(len(l)):
            K[Qj - len(l) + i, j] = l[i]    #对k每列初始化赋值，每列都不一样，一列代表一种知识点掌握情况
    std = np.sum(Q, axis=1)    #axis=1按照x轴方向计算，对Q知识点考察矩阵进行累计加。一个题目考察多少个知识点。
    r = (Q * K == std) * 1    #r矩阵表示理论上j这道题对于l这个模式能否做对 所以R为潜在作答矩阵
    sg = 0.01 * np.ones((nj, 2))       #sg初始化为nj行，2列  二维数组，每组为一个sg

    continueSG = True
    kk =1      #迭代次数
    lastLX = 1
    # count iteration times
    # student*pattern = student* problem       problem*skill         skill*pattern
    #开始 E-M step
    #aPrior = np.ones(2 ** Qj) / 10 ** 8  # math2
    while continueSG == True:
        # E step，calculate likelihood matrix
        IL = np.zeros((ni, 2 ** Qj))  #IL训练集学生所有技能模式的似然概率矩阵 技能模式的数量482人，每个人256种可能性
        IR = np.zeros((4, nj))
        # skill pattern number
        if multi==True:
            #print('mult    i 4 processes')
            with Pool(processes=4) as pool:
                multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, Qj, i)) for i in range(4)]
                for item in ([res.get(timeout=1000) for res in multiple_results]):
                    IL += item

                sumIL = IL.sum(axis=1)
                LX = np.sum([i for i in map(math.log2, sumIL)])
                #print('LX', LX)

                IL = (IL.T / sumIL).T * aPrior

                multiple_results = [pool.apply_async(MStep, (IL, n, r, Qj, i)) for i in range(4)]
                for item in ([res.get(timeout=1000) for res in multiple_results]):
                    IR += item
        else:
            #print('single process')
            for l in range(2 ** Qj):
                lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
                    1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
                IL[:, l] = lll.prod(axis=1) #IL训练集学生所有技能模式的似然概率矩阵
            sumIL = IL.sum(axis=1)
            LX = np.sum([i for i in map(math.log2, sumIL)])
            #print('LX', LX)
            IL = (IL.T / sumIL).T* aPrior  #似然矩阵IL
            n1 = np.ones(n.shape)
            for l in range(2 ** Qj):   #根据四个变量 更新sg的值，并重新计算似然矩阵IL
                IR[0] += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
                IR[1] += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
                IR[2] += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
                IR[3] += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
        if abs(LX-lastLX)<threshold: # 超参1
            continueSG = False
        lastLX = LX
        sg[:,1] = IR[1] / IR[0]            #g猜测率 guess
        sg[:,0] = (IR[2]-IR[3]) / IR[2]   # s 失误率 slip
        #print('[%s]次迭代 [%s]名学生 [%s]个问题 '%(kk,ni,nj))
        kk +=1   #kk为迭代次数，ni为学生人数，nj为问题个数
    print(kk)
    endTime = time.time()
    #print('DINA训练时间为 :[%.3f] s'%(endTime-startTime))

    for i in range(Qi):
        if sg[i, 0] > threshold_slip:
           sg[i, 0] = threshold_slip
    for j in range(Qi):
        if sg[j, 1] > threshold_guess:
            sg[j, 1] = threshold_guess


    return sg,r   #r矩阵表示理论上j这道题对于l这个模式能否做对

def trainIDINAModel(n,Q,threshold):
    startTime = time.time()
    #print('training IDINA model')
    ni, nj = n.shape
    Qi, Qj = Q.shape
    sg = np.zeros((nj, 2))
    k = Qj
    K = np.mat(np.zeros((k, 2 ** k), dtype=int))
    for j in range(2 ** k):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[k - len(l) + i, j] = l[i]
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1
    for i in range(nj):
        sg[i][0] = 0.01
        sg[i][1] = 0.01
    continueSG = True
    kk =1
    IL = np.ones((ni, 2 ** Qj))
    istart = 0
    istop = ni
    while continueSG == True:
        for i in range(istart,istop):
            IL[i] = 1
            lll = ((1 - sg[:, 0]) ** n[i] * sg[:, 0] ** (1 - n[i])) ** r.T.A * (sg[:, 1] ** n[i] * (
            1 - sg[:, 1]) ** (1 - n[i])) ** (1 - r.T.A)
            IL[i] = lll.prod(axis=1)
        istart = istop % ni    #1   11  21   （只更新istart到istop之间的似然函数值）
        istop = istart + 10    #11  21  31
        if istop > ni:
            istop = ni
        I0 = np.zeros(nj)
        R0 = np.zeros(nj)
        I1 = np.zeros(nj)
        R1 = np.zeros(nj)
        n1 = np.ones(n.shape)
        for l in range(2 ** Qj):
            I1 += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
            R1 += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
            I0 += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
            R0 += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
        if (abs(R0 / I0 - sg[:, 1]) < threshold).any() and (abs((I1 - R1) / I1 - sg[:, 0]) < threshold).any():
            #RO/IO为猜测率
            continueSG = False
        sg[:, 1] = R0 / I0
        sg[:, 0] = (I1 - R1) / I1
        #print(sg)
        #print('[%s] time [%s] students [%s] problems'%(kk,ni,ni))
        kk += 1
    endTime = time.time()
    #print('IDINA model cost time: [%.3f] s'%(endTime-startTime))
    for i in range(Qi):
     if sg[i, 0] > threshold_slip:
       sg[i, 0] = threshold_slip
    for j in range(Qi):
       if sg[j, 1] > threshold_guess:
        sg[j, 1] = threshold_guess
    return sg,r

def continuously(IL):
    ni,nj = IL.shape
    Qj = (int)(math.log2(nj))
    continuous = np.ones((ni, Qj))
    denominator = np.sum(IL, axis=1)
    for j in range(Qj):
        molecule = np.zeros(ni)
        for l in range(nj):
            ll = list(bin(l).replace('0b', ''))
            if j < len(ll) and ll[len(ll) - j - 1] == '1':
                molecule += IL[:, l]
        continuous[:, Qj - 1 - j] = molecule / denominator
    return continuous

def discrete(continuous):
    ni,k = continuous.shape
    a = np.zeros(ni,dtype=int)
    for i in range(ni):
        for ki in range(k):
            if continuous[i][ki]>0.5:
                a[i] += 2**(k-ki-1)
    return a

def predictDINA(n,Q,sg,r,):
    startTime = time.time()
    #print('---------------开始预测---------------')
    ni, nj = n.shape
    Qi, Qj = Q.shape  #20道题，分为8个知识点
    IL = np.zeros((ni, 2**Qj))  #传入的测试集学生的256个不同模式矩阵
    if multi == True:
        #print('multi 4 processes')
        with Pool(processes=4) as pool:
            multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, Qj, i)) for i in range(4)]
            for item in ([res.get(timeout=1000) for res in multiple_results]):
                IL += item
    else:
        for l in range(2 ** Qj): #l为学生的所有可能模式
            lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
                1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
            IL[:, l] = lll.prod(axis=1)    #此处的LLL为用测试数据得到的数据，IL训练集学生所有技能模式的似然概率矩阵
    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    a = IL.argmax(axis=1)  #a为最大似然 axis=1  x轴方向查找最大值的索引下标 ，a为一维数组其中存储每位学生的最大l模式的索引下标
    unique, counts = np.unique(a, return_counts=True) #unique函数去除其中重复的元素，并按元素由小到大返回一个新的无元素重复存储在unique中。
    aPrior[unique] = counts/len(a)                    #counts中存放新列表元素在旧列表中的位置，并以列表形式储存在counts中。
    K = np.mat(np.zeros((Qj, 2 ** Qj), dtype=int))
    for j in range(2 ** Qj):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[Qj - len(l) + i, j] = l[i]
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1  #Q为 j*k矩阵，
    i, j = n.shape      #n.T 将矩阵n转置  n为测试集
    p = np.sum((r[:,a] == n.T) * 1) / (i * j)  #i为学生数，j为题目数
    #print('总共有 [%s] 个人, 精准度为 [%.3f]'%(ni, p))
    #print('predict time [%.3f] s' %(time.time() - startTime))
    #p1 = np.sum((r[:, a] == n.T) * 1) / (i * j)  # n是实际答题情况 a是算出来的学生知识点掌握
    rrrSum = []
    nnnSum = []
    for rrr, nnn in zip(r[:, a], n.T):
        rrrSum.extend(rrr.tolist()[0])
        nnnSum.extend(nnn)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    #print(len(nnnSum), nnnSum)
    #print(len(rrrSum), rrrSum)
    C2 = confusion_matrix(nnnSum, rrrSum, labels=[0, 1])
    TP = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    TN = C2[1][1]
    Precision = TP / (TP + FP)  # 查准率或精准率
    Recall = TP / (TP + FN)     # 召回率或查全率
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    AUC = roc_auc_score(nnnSum, rrrSum)
    #Accuracy = p1
    Accuracy =(TP + TN) / (TP+TN+FP+FN)
    #print('测试：accuracy=[%s]  precision=[%s] recall=[%s]  f1=[%s]  auc=[%s]  ' % (
    #Accuracy, Precision, Recall, F1, AUC))
    return Accuracy, Precision, Recall, F1, AUC




def trainAndPredict(model,dataSet,threshold):
    print('模型:[%s]   数据集:[%s]' %(model,dataSet))
    if dataSet == 'FrcSub':
        n = pd.read_csv('math2020/FrcSub/data.csv').values    #学生试题得分矩阵X，这里为n
        Q = np.mat(pd.read_csv('math2020/FrcSub/q.csv'))      #知识点考察矩阵Q
    elif dataSet == 'Math1':
        n = pd.read_csv('math2020/Math1/data.csv').values
        Q = np.mat(pd.read_csv('math2020/Math1/q.csv').head(15).values)
    elif dataSet == 'Math2':
        n = pd.read_csv('math2020/Math2/data.csv').values
        Q = np.mat(pd.read_csv('math2020/Math2/q.csv'))
    elif dataSet == 'Math_DMiC_1':
        n = pd.read_csv('math2020/Math_DMiC_1/data1.csv').values
        Q = np.mat(pd.read_csv('math2020/Math_DMiC_1/q0_1.csv'))
    elif dataSet == 'Math_DMiC_2':
        n = pd.read_csv('math2020/Math_DMiC_2/data1.csv').head(500).values
        Q = np.mat(pd.read_csv('math2020/Math_DMiC_2/q1_1.csv'))
    elif dataSet == 'Math_DMiC_3':
        n = pd.read_csv('math2020/Math_DMiC_3/data1.csv').head(500).values
        Q = np.mat(pd.read_csv('math2020/Math_DMiC_3/q2_1.csv'))
    elif dataSet == 'test':
        n = pd.read_csv('math2020/test/data.csv').values
        Q = np.mat(pd.read_csv('math2020/test/q.csv'))

    else:
        print('dataSet not exist!')
        exit(0)

    #n cross verify
    n_splits = 5
    KF = KFold(n_splits=n_splits,shuffle=False)
    #KF = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    #交叉验证函数，分10折，然后随机打乱，得到
    precision = 0
    s1=0
    s2=0
    s3=0
    s4=0
    s5=0
   # flag=0
    for train_index, test_index in KF.split(n):   #train_index训练集   test为测试集  n为学生得分矩阵
        X_train, X_test = n[train_index], n[test_index]
        print("train student: %s" % (len(X_train)))
        print("test student : %s" % (len(X_test)))
        #Write_data(X_train, flag)
        #flag = flag + 1
        #Write_data(X_test,flag)
        #flag =flag +1

        if model == 'DINA':
            sg,r = trainDINAModel(X_train,Q,threshold)
            #print(r)
        else:
            sg,r = trainIDINAModel(X_train,Q,threshold)     #r矩阵表示理论上j这道题对于l这个模式能否做对（潜在能力矩阵）
        #try:
        Accuracy, Precision, Recall, F1, AUC =predictDINA(X_test, Q, sg, r,)
        #except ValueError:
               #  pass
        s1 +=Accuracy
        s2 +=Precision
        s3 +=Recall
        s4 +=F1
        s5 +=AUC
    s1 = s1/n_splits
    s2 = s2/n_splits
    s3 = s3/n_splits
    s4 = s4/n_splits
    s5 = s5/n_splits
    print('测试:accuracy=[%s]  precision=[%s] recall=[%s]  f1=[%s]  auc=[%s] ' %(s1,s2,s3,s4,s5))
    return s1, s2,s3,s4,s5

def testPredict(model, nName, qName, threshold):
    n = pd.read_csv(nName).values  # 学生-试题
    q = np.mat(pd.read_csv(qName))  # 试题 - 知识点
    nx, ny =n.shape
    qx, qy =q.shape
    with open('./recordTestPrec.txt','a') as f:
        f.writelines('数据集大小为：'+str(nx)+ '*'+ str(ny)+'*'+str(qy)+'\n')
    #KF = KFold(n_splits=5, shuffle=False)
    KF = KFold(n_splits=5,shuffle=True,random_state=1)
    KFtimes =1
    precision = 0
    for train_index ,test_index in KF.split(n):
        X_train,X_test = n[train_index] , n[test_index]
        print("train student: %s" %(X_train))
        print("test student : %s" %(X_test))
        if model == 'DINA':
            sg,r = trainDINAModel(X_train,q,threshold)
        else :
            sg,r = trainIDINAModel(X_train,q,threshold)
        Accuracy, Precision, Recall, F1, AUC = predictDINA(X_test, q, sg, r)
        precision +=Precision
        KFtimes+=1
    print("准确率平均值为：")
    averp =precision / 5
    print(averp)
    with open('./recordTestPrec.txt','a') as f:
        f.writelines('\n最终准确率为'+str(precision/5))
    return averp




def cv_precision( threshold):

    precision = testPredict('DINA','math2020/FrcSub/data.csv','math2020/FrcSub/q.csv',threshold)
    precision = np.array(precision)
    return precision


def main():
    startTime = time.time()
    global  multi, aPrior,threshold

    threshold = 0.01   #超参

    multi = False
    #aPrior = np.ones(2 ** 8) / 10 ** 8   #FruSub
    #aPrior = np.ones(2 ** 11) / 10 ** 8     #math1
    aPrior = np.ones(2 ** 16) / 10 ** 8  # math2
    #aPrior = np.ones(2 ** 23) / 10 ** 8  # FruSub
    dataSet = ('FrcSub', 'Math1', 'Math2', 'Math_DMiC_1','Math_DMiC_2','Math_DMiC_3','test')
    model = ('DINA', 'IDINA')
    s1=0
    s2=0
    s3=0
    s4=0
    s5=0
    for js in range(0,1):
        Accuracy_1, Precision_1, Recall_1, F1_1, AUC_1 = trainAndPredict(model[0], dataSet[2], threshold)
        s1 += Accuracy_1
        s2 += Precision_1
        s3 += Recall_1
        s4 += F1_1
        s5 += AUC_1
    accuracy_average = s1 / 3
    precision_average = s2 / 3
    recall_average = s3 /3
    f1_average = s4 / 3
    auc_average = s5 / 3

    '''bds = {'threshold': (1,500)}

    rf_bo = BayesianOptimization(
        cv_precision,
        bds
    )
    rf_bo.maximize()

    #rf_bo.res['max'] '''
    print('---------数据集：[%s]运行后的数据------' % (dataSet[2]))
    print('main_accuracy=[%.6f]  main_precision=[%.6f] main_recall=[%.6f]  mean_f1=[%.6f]  mean_auc=[%.6f]  ' % (
        accuracy_average, precision_average, recall_average, f1_average, auc_average))
    print('总 耗 时 :[%.3f] s' %(time.time()-startTime))


if __name__ == "__main__":
    main()
