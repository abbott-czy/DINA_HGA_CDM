from myAlgorithms import *
from multiprocessing import Pool



def testModel(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, slip, guess, data_patch_id, run_id=1,
              n_pop=50, max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):
    if n_knowledge_coarse == n_knowledge_fine:
        multi = False

        Accuracy, Precision, Recall, F1, AUC = predictDINA(data, q_matrix, slip, guess, multi,data_patch_id, run_id=1)
    else:
        Accuracy, Precision, Recall, F1, AUC = testModel_EA(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, slip, guess, data_patch_id, run_id,
              n_pop, max_generations, alg_name, data_name)
    # multi = False
    # Accuracy, Precision, Recall, F1, AUC = testModel_EA(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, slip, guess, data_patch_id, run_id,
    #           n_pop, max_generations, alg_name, data_name)
    return Accuracy, Precision, Recall, F1, AUC


def predictDINA(data, q_matrix, slip, guess, multi,data_patch_id, run_id=1):

    startTime = time.time()
    slip = np.array(slip)
    guess = np.array(guess)
    data = np.array(data)
    q_matrix = np.mat(q_matrix)
    n_questions, n_knowledge = q_matrix.shape

    # crate K matrix，indict k skill could get how many vector
    # 构造K矩阵，表示k个技能可以组成的技能模式矩阵
    k_matrix = np.mat(np.zeros((n_knowledge, 2 ** n_knowledge), dtype=int))  # k行 8行 、 2**k列 256列
    for j in range(2 ** n_knowledge):  # 0到255 共256个数
        l = list(bin(j).replace('0b', ''))  # 范围是0到255的二进制 0000 0000-1111 1111
        for i in range(len(l)):  # 0到7 l的长度为8
            k_matrix[n_knowledge - len(l) + i, j] = l[i]  # 8行 256列 的所有可能的技能矩阵赋值完毕


    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    std = np.sum(q_matrix, axis=1)  # 将每一行的元素相加,将Q矩阵压缩为一列 20行1列
    '''
    # rer= Q * K 20*256 
    第一行 0 0 1 1 1 1 2 2 0 0 - 1 1 1 1 2 2 1 1 2 2 - 2 2 3 3 1 1 2 2 2 2 - ...前30个共256
    # rerb = (Q * K == std)  # 20*256 false or True
    第一行 
    False False False False False False False False False False 
    False False False False False False False False False False 
    False False  True  True False False False False False False...前30个共256
    '''
    r_matrix = (q_matrix * k_matrix == std) * 1  # Q:20x8 K:8x256  r:20x256
    # r后面保持不变

    print('训练结束，预测开始')

    ni, nj = data.shape  # 108*20 五折中测试折108学生 学生作答矩阵
    Qi, Qj = q_matrix.shape  # 20*8  知识点考察矩阵
    # 预测的每个学生的技能向量
    IL = np.zeros((ni, 2 ** Qj))  # 428*256 似然函数矩阵
    k = Qj  # 8个知识点

    if multi == True:
        print('预测 multi 4 processes')
        with Pool(processes=4) as pool:
            multiple_results = [pool.apply_async(EStep_multi, (IL, slip, guess, data, r_matrix, k, i)) for i in range(4)]
            for item in ([res.get(timeout=1000) for res in multiple_results]):
                IL += item
        # 计算得到IL 似然函数矩阵
    else:
        for l in range(2 ** Qj):
            # 学生的数量
            lll = ((1 - slip) ** data * slip ** (1 - data)) ** r_matrix.T.A[l] * (guess ** data * (
                1 - guess) ** (1 - data)) ** (1 - r_matrix.T.A[l])
            IL[:, l] = lll.prod(axis=1)

    # 只需要在上面的IL中，针对每一个学生，寻找他在所有l模式中，似然函数最大的那一项l
    a = IL.argmax(axis=1)  #IL中每行的最大值的索引 a:108*1 这是一折
    # 调试*调试*调试*调试 看看学生的知识点掌握 *调试*调试*
    # ai = np.size(a,0)  # 学生数
    # print('测试集学生数：'+str(ai))
    # for i in range(ai):
    #     tem=bin(a[i]).replace('0b', '')
        # tem2=tem.zfill(Qj)  # 填充到知识点位数
        # print('学生'+str(i)+'的知识点掌握情况')
        # print(tem2)
    # *调试*调试*调试*

    # a2 = discrete(continuously(IL))
    # print('连续化向量')
    # print(continuous)

    # 计算准确率
    # Q:20题x8点 K:8点x256  r:20题x256   a:学生数*1  n:学生数*题目
    i, j = data.shape  # 108行学生*20列题目
    # print('总共有' + str(ni) + '个人，a准确率为：')
    # print("r[:, a]",list(r[:, a]),len(list(r[:, a])[0]))
    # print("n.T",list(n.T),len(list(n.T)[0]))
    # print("a",a)
    p1 = np.sum((r_matrix[:, a] == data.T) * 1) / (i * j)   # n是实际答题情况 a是算出来的学生知识点掌握
    rrrSum = []
    nnnSum = []
    for rrr, nnn in zip(r_matrix[:, a], data.T):
        rrrSum.extend(rrr.tolist()[0])  # tolist函数，将矩阵转化为列表  #rrrSum为预测的值 ，
        nnnSum.extend(nnn)   # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。nnnSum原学生的数据

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    print(len(nnnSum),nnnSum)
    print(len(rrrSum),rrrSum)
    #  erratum=rrrSum.T
    #write_rrrSum(erratum,data_patch_id, run_id)
    C2 = confusion_matrix(nnnSum, rrrSum, labels=[0, 1])  # nnn为真实的作答矩阵，rrr为预测的作答情况，label给出的类别，
    TP = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    TN = C2[1][1]
    # TODO：确认原代码是否有错
    # TP = C2[0][0]
    # FP = C2[0][1]
    # FN = C2[1][0]
    # TN = C2[1][1]
    Precision = TP / (TP + FP)  # 查准率或精准率  查准率
    Recall = TP / (TP + FN)  # 召回率或查全率      查全率
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    AUC = roc_auc_score(nnnSum, rrrSum)
    # Accuracy = p1
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # TODO
    #Accuracy = p1#fit_max / (n_students * n_questions)


    # with open('./recordTestPrec.txt','a') as f:
    #     f.writelines('该折准确率'+str(p1))

    # 连续化向量准确率
    # print('总共有' + str(ni) + '个人，a2准确率为：')
    # p1 = np.sum((r[:, a2] == n.T) * 1) / (i * j)
    # print(p1)

    print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    print('------------------预测结束-----------------------------')
    return Accuracy, Precision, Recall, F1, AUC


def testModel_EA(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, slip, guess, data_patch_id, run_id=1,
              n_pop=50, max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):

    startTime = time.time()
    print("预测为EA model")
    flag_train = False
    n_knowledge = n_knowledge_fine
    if alg_name == 'GA':
        resultPop, logbook, slip, guess = GA(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data,
                                                 q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                 flag_train, max_generations, len_s_g)
    elif alg_name == 'GA_NBC' or alg_name == 'GA_NBC_multi':
        resultPop, logbook, slip, guess = GA_NBC(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data,
                                             q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                             flag_train, max_generations, len_s_g, data_name)
    # 输出最优解
    index = np.argmax([ind.fitness for ind in resultPop])
    # S,G = decode(resultPop[index])
    # print('当前最优解：' + str(resultPop[index]) + '\t对应的函数值为：' + str(resultPop[index].fitness))
    # fit_maxs = logbook.select('max')
    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]                     
    #Accuracy = fit_max / (n_students * n_questions)  # 准确率计算

    # 计算混淆矩阵
    label = []
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    for i in range(n_students):
        for j in range(n_questions):
            label.append(data[i][j])
    bestIndividual = list(resultPop[index])
    # 从生成的种子中提取随机生成的学生*知识点掌握情况矩阵A
    bestA = acquireA(bestIndividual, n_students, n_knowledge)
    # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量
    bestYITA = acquireYITA(bestA, q_matrix, n_students, n_questions, n_knowledge)
    # 位于func文件中 102行 X为引入s,g后的作答得分情况矩阵 , Xscore实际矩阵
    bestX, Xscore = acquireX(n_students, n_questions, bestYITA, slip, guess) #
    predict = []
    for i in range(n_students):
        for j in range(n_questions):
            predict.append(bestX[i][j])
    predictScore = []
    for i in range(n_students):
        for j in range(n_questions):
            predictScore.append(Xscore[i][j])
    # print(len(label))
    # print(len(predict))  # 最佳个体的基因长度 即student*knowlege 216=24学生*9知识点
    C2 = confusion_matrix(label, predict, labels=[0, 1])  # label为学生作答矩阵 ， predict 预测矩阵
    print(len(label))
    print(label)
    print(len(predict))
    print(predict)
    TP = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    TN = C2[1][1]
    # TODO：确认原代码是否有错
    # TP = C2[0][0]
    # FP = C2[0][1]
    # FN = C2[1][0]
    # TN = C2[1][1]
    Precision = TP / (TP + FP)  # 查准率或精准率  查准率
    Recall = TP / (TP + FN)  # 召回率或查全率      查全率
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    AUC = roc_auc_score(label, predict)
    Accuracy = fit_max / (n_students * n_questions)
    #Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # TODO
    # Accuracy = p1#fit_max / (n_students * n_questions)

    # with open('./recordTestPrec.txt','a') as f:
    #     f.writelines('该折准确率'+str(p1))

    # 连续化向量准确率
    # print('总共有' + str(ni) + '个人，a2准确率为：')
    # p1 = np.sum((r[:, a2] == n.T) * 1) / (i * j)
    # print(p1)

    print('预测消耗时间：' + str(int(time.time()) - int(startTime)) + '秒')
    print('------------------预测结束-----------------------------')
    return Accuracy, Precision, Recall, F1, AUC


def EStep_multi(IL, slip, guess, n, r, k, i):
    base = 2**(k-2)  # k=8 base=64
    for l in range(i*base,(i+1)*base):  #0*64-1*64、1*64-2*64、2*64-3*64、3*64-4*64 一共256
        # 学生的数量
        lll = ((1 - slip) ** n * slip ** (1 - n)) ** r.T.A[l] * (guess ** n * (
            1 - guess) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)  # lll中一行元素连乘，形成一列，赋给IL的第l列 共256列 for循环256次 赋值256次
    return IL




'''
输入：IL,n,r,k,i
输出：IR
'''
def MStep_multi(IL,n,r,k,i):
    base = 2**(k-2)  # k=8 base=64
    ni,nj=n.shape
    IR = np.zeros((4, nj))
    n1 = np.ones(n.shape)
    for l in range(i*base,(i+1)*base):
        IR[0] += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
        IR[1] += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
        IR[2] += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
        IR[3] += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
    return IR

