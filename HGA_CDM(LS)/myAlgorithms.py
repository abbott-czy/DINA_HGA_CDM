import copy
import math
import random
from queue import Queue
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.metrics import roc_auc_score

from tools import *
import os


# testModel.py 和 train_model.py引用此库

def GA_NBC(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess, data_patch_id,
           run_id, n_pop,
           flag_train, max_generations, len_s_g, alg_name='GA_NBC', data_name='Math_DMiC'):
    # 传入参数：alg_name：算法， data_name：数据集，学生作答矩阵data，知识考察矩阵q_matrix，max_runs：多进程参数
    # max_split 分折数， max_generations=遗传代数 ， n_pop ：种群人数 , is_multi：是否进行多进程，data为训练集人数。
    # n_knowledge_coarse 知识点粒度数  ， n_knowledge_fine 传入的知识点数   flag_train bool型 是否为训练 传入 true
    # i为第几次训练和测试 一共五次
    n_knowledge = n_knowledge_fine
    # 优化目标：单变量，求weights=1：最大值，weights=-1 为最小值
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # 创建Individual类，继承list，求最大值 初始化基因编码位数,初始值,等基本信息
    creator.create('Individual', list, fitness=creator.FitnessMax)
    # 基因长度
    if flag_train:
        GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g  # 32*11+15*2*7
        # 基因长度等于：I*K + J*2* jingdu  I*k代表学生对知识点的认知情况 后面是每道题对应一个失误率和猜测率，并且精度为len_s_g
    else:
        GENE_LENGTH = n_students * n_knowledge  # 学生数8*知识点数11

    toolbox = base.Toolbox()  # Operators都在toolbox模块里，要给每个算子选择合适算法，一个工具箱
    # 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
    toolbox.register('Binary', bernoulli.rvs, 0.5)  # 等概率取0和1
    # 注册用tools.initRepeat生成长度为GENE_LENGTH的Individual、等概率取0和1、个体的基因长度
    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)
    # 注册生成群落的函数
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    # 生成初始族群
    pop = toolbox.Population(n=n_pop)

    # 定义评价函数
    def evaluate(individual):  # 评价或进化(Evaluation): 设定一定的准则评价族群内每个个体的优秀程度。这种优秀程度通常称为适应度(Fitness)。
        # 从生成的种子中提取随机生成的学生*知识点掌握情况矩阵A
        A = acquireA(individual, n_students, n_knowledge)

        # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量
        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)  # func.py的67行 ，
        slip_ratio = slip
        guess_ratio = guess
        if flag_train:
            # 此处的S和G是数字形式，不是0101形式
            s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)  # 列表 大小为question，func79行
            # 获取存储所有s值的列表和存储所有g值的列表
            slip_ratio = s
            guess_ratio = g

        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)
        # rrrSum = []
        # nnnSum = []
        # for rrr, nnn in zip(X, data):
        #     rrrSum.extend(rrr)  # tolist函数，将矩阵转化为列表  #rrrSum为预测的值 ，
        #     nnnSum.extend(nnn)  # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。nnnSum原学生的数据
        #
        # AUC = roc_auc_score(nnnSum, rrrSum)
        # return (AUC),
        sum_ = 0
        for i in range(n_students):
            for j in range(n_questions):
                sum_ = sum_ + (1 - abs(data[i][j] - X[i][j]))  # 符合则sum+1 不符则+0

        return (sum_),  # 返回的sum越大说明data与X越匹配

    # 在工具箱中注册遗传算法需要的工具
    toolbox.register('evaluate', evaluate)
    # 注册Tournsize为2的锦标赛选择 ，每次从总体中取出两个个体，选择最优的一个保留至下一代
    toolbox.register('select', tools.selTournament, tournsize=3)
    # cxUniform为均匀交叉，交叉概率为0.5
    toolbox.register('mate', tools.cxUniform, indpb=0.5)  # 注意这里的indpb需要显示给出
    # mutFlipBit 为位翻转突变：对个体中的每一个基因按给定对变异概率取非。概率位0.3 ，超参indpb需要注意
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.5)

    # 注册计算过程中需要记录的数据 也是数据记录
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # 计算适应度的函数
    stats.register("avg", np.mean)  # 平均值
    stats.register("std", np.std)  # 标准差
    stats.register("min", np.min)  # 最小值
    stats.register("max", np.max)  # 最大值

    # 注册记录
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields  # 表头gen nevals avg std min max

    # 评价族群
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]  # 无效价值的个体列表
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # 计算他们各自的适应度 适应度列表
    # for ind in pop:
    #     ind = local_search_knowledge(data, ind, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
    #                                  len_s_g)
    #     ind.fitness.values = evaluate(ind)
    #     print(ind.fitness.values)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit  # 赋值适应度
        # print(fit)

    # 根据族群适应度，编译出stats记录
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)  # 第0代
    tempTotalPop = []
    LSmax = 100  # 限制每个时刻局部搜索的最大次数
    Nls = 0
    # 下面开始迭代！！！
    LSMcm = []  # 存放每个时刻进行局部搜索后的个体和他的影响半径
    LSMdis = []
    LSM_bestFitness_pre = []
    LSM_bestFitness_post = []
    for gen in range(1, max_generations + 1):  # 训练代数全局变量 方便输出数据好看

        # print(str(run_id) + '当前第' + str(gen) + "代")
        # if gen % 50 == 0:
        #     print('当前第' + str(gen) + "代")
        if (gen == 1):
            totalPop = invalid_ind  # 无效价值的个体列表
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)  # 汉明距离二维列表
        # print("开始NBC算法"+str(computeTime()))
        try:
            newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)  # NBC算法后 func 137行
        except:
            pass
        # print("newg中个体数量: ", len(newg))
        # 对所有类当中的每一类
        # 进行循环交叉和突变
        Popth = 0
        for PopNodeList in newg:
            local_flag = True
            Popth += 1
            # print(str(Popth))
            # print("第"+str(Popth)+"个个体计算" + str(computeTime()))
            subPop = []

            for nodeId in PopNodeList.vs['label']:
                try:# vs['label']图的顶点，既各个个体染色体
                    subPop.append(totalPop[nodeId])  # 选择一些个体
                except :
                    pass


            # cxpb = (0.9 - ((0.9 - 0.3) / (max_generations)) * gen)  # 交叉率
            cxpb = 0.4329
            # mutpb = (0.9 - ((0.9 - 0.3) / (max_generations)) * gen)  # 变异率
            mutpb = 0.09351
            # TODO
            # n_i = n_students * n_knowlege
            # ind_tmp = subPop[0]
            # ind = np.array(ind_tmp[0:n_i]).reshape((n_students, n_knowlege))
            # for i in range(n_students):
            #     np.where(ind[i] == 1)
            N_subPop = len(subPop)
            ind_seed = subPop[0]  #

            # oldfitness = evaluate(ind)
            # 局部搜索,找局部最优
            seed_LS = toolbox.clone(ind_seed)
            seed_LS_tem = toolbox.clone(ind_seed)
            if alg_name == 'HGA_LS':
                if flag_train:  # 训练集
                    if n_knowledge_coarse == n_knowledge_fine:
                        seed_LS = local_search_train_0(data, seed_LS, q_matrix, n_students, n_knowledge_fine,
                                                       n_questions, GENE_LENGTH, len_s_g)  # 524行，返回更新后的个体
                    else:
                        seed_LS = local_search_train(data, seed_LS, q_matrix, n_students, n_knowledge_coarse,
                                                     # 527行
                                                     n_knowledge_fine, n_questions, GENE_LENGTH, len_s_g,
                                                     data_name)
                else:  # 测试集合
                    if n_knowledge_coarse < n_knowledge_fine:
                        seed_LS = local_search_test(data, seed_LS, q_matrix, n_students, n_questions,
                                                    n_knowledge_coarse,
                                                    n_knowledge_fine, slip, guess, data_name)
                    else:
                        seed_LS = local_search_test_0(data, seed_LS, q_matrix, n_students, n_knowledge, slip,
                                                      guess)
                seed_LS.fitness.values = evaluate(seed_LS)
            if alg_name == 'GA_NBC' :
                PMatrix = hammingDis(subPop)  # 得到P种子所在种族的所有距离
                PRadius_tem = []
                for i in PMatrix:
                    PRadius_tem.append(max(i))
                PRadius = max(PRadius_tem)
                if Popth == 1 and gen==1:
                    local_flag = True
                    # and distance(seed_LS, i) < 50
                else:
                    if Nls > LSmax:
                        local_flag = False
                    elif int(seed_LS.fitness.values[0]) < max(LSM_bestFitness_pre):
                        local_flag = False
                    elif int(seed_LS.fitness.values[0]) > max(LSM_bestFitness_post):
                        local_flag = True
                    else:
                        for  i   in LSMcm:
                            if distance(seed_LS,i) < LSMdis[LSMcm.index(i)] :
                                local_flag = False
                        # for i in LSMcm:  # toolbox.clone保证我们拥有独立的子代，而不是只保存子代的引用。 LSMcm中存放这局部搜索的记忆体
                        #     for j in LSMdis:  # LSMdis 存放这各自的距离影响半径
                        #         if distance(seed_LS, i) < j:
                        #             local_flag = False

                if local_flag and Nls < LSmax:
                    Nls = Nls + 1
                    LSMcm.append(seed_LS_tem)
                    if flag_train:  # 训练集
                        if n_knowledge_coarse == n_knowledge_fine:
                            seed_LS = local_search_train_0(data, seed_LS, q_matrix, n_students, n_knowledge_fine,
                                                           n_questions, GENE_LENGTH, len_s_g)  # 524行，返回更新后的个体
                        else:
                            seed_LS = local_search_train(data, seed_LS, q_matrix, n_students, n_knowledge_coarse,
                                                         # 527行
                                                         n_knowledge_fine, n_questions, GENE_LENGTH, len_s_g,
                                                         data_name)
                    else:  # 测试集合
                        if n_knowledge_coarse < n_knowledge_fine:
                            seed_LS = local_search_test(data, seed_LS, q_matrix, n_students, n_questions,
                                                        n_knowledge_coarse,
                                                        n_knowledge_fine, slip, guess, data_name)
                        else:
                            seed_LS = local_search_test_0(data, seed_LS, q_matrix, n_students, n_knowledge, slip,
                                                          guess)
                    seed_LS.fitness.values = evaluate(seed_LS)

                    tem_distance = distance(seed_LS_tem, seed_LS)
                    # print("tem_distance is :")
                    # print(tem_distance)
                    tem = max(tem_distance, PRadius)
                    LSMdis.append(tem)
                    LSMcm.append(seed_LS)
                    LSMdis.append(tem)
                    if len(LSM_bestFitness_post) == 0:
                        LSM_bestFitness_post.append(int(seed_LS.fitness.values[0]))
                    elif max(LSM_bestFitness_post) < int(seed_LS.fitness.values[0]):
                        LSM_bestFitness_post.append(int(seed_LS.fitness.values[0]))
                    if len(LSM_bestFitness_pre) == 0:
                        LSM_bestFitness_pre.append(int(seed_LS_tem.fitness.values[0]))
                    elif max(LSM_bestFitness_pre) < int(seed_LS_tem.fitness.values[0]):
                        LSM_bestFitness_pre.append(int(seed_LS_tem.fitness.values[0]))

            # newfitness = evaluate(newind)

            # 配种选择
            offspring = toolbox.select(subPop, N_subPop)  # 后代，进行选择配种，适应度最好的留下
            # 一定要复制，否则在交叉和突变这样的原位操作中，会改变所有select出来的同个体副本
            offspring_Xor = []  # [toolbox.clone(ind) for ind in offspring]
            for i in range(N_subPop):
                offspring_Xor.append(copy.deepcopy(offspring[i]))
                offspring_Xor.append(copy.deepcopy(seed_LS))

            # 变异操作 - 交叉     所有后代中相邻两个个体一定几率交叉
            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
                if random.random() < cxpb:  # random.random()生成0和1之间的随机浮点数float
                    toolbox.mate(child1, child2)  # mate函数，进行交叉
                    del child1.fitness.values
                    del child2.fitness.values

            # 评价当前没有fitness的个体，确保整个族群中的个体都有对应的适应度
            invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # evaluate 选择评价函数 ，给没有适应度的交叉个体赋予适应度
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 变异操作 - 突变
            offspring_mut = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                if random.random() < mutpb:  # mutpb突变率
                    toolbox.mutate(mutant)  #
                    del mutant.fitness.values

            # 评价当前没有fitness的个体，确保整个族群中的个体都有对应的适应度
            invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # evaluate 选择评价函数 ，给没有适应度的交叉个体赋予适应度
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = offspring_Xor + offspring_mut + subPop
            offspring.append(seed_LS)
            # TODO: 选择一部分精英，再随机化一部分个体

            # 环境选择 - 保留精英
            # 对整个子种群进行排序
            offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')  # 选择精英,保持种群规模len(offspring)
            # pop_selected = Queue.Queue()
            pop_selected = []
            pop_selected.append(offspring[0])
            num_selected = 1
            # pop_selected.pop()
            for i in range(1, len(offspring)):
                idx = pop_selected[num_selected - 1]
                dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
                if dis >= 5:
                    pop_selected.append(offspring[i])
                    num_selected = num_selected + 1
            pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')
            # 剔除冗余个体

            # idx = 0
            # for i in range(1, len(offspring)):
            # offspring(idx,:)
            # cmp(offspring(idx,:), offspring(i,:))

            tempTotalPop.extend(pop_selected)  # extend函数，在tempTotalPop的末尾，一次性加上pop_selected

        # print("LSMdis里的距离为：")
        # print(LSMdis)
        # print("每代局部搜索的次数： ")
        # print(LS)
        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)  # 每一代记录一下
    # print(len(LSMcm))
    # print(LSM_bestFitness_post)
    # print(LSM_bestFitness_pre)
    with open('./czy_results/ALS/als.txt', 'a') as f:
        f.write(str(Nls) + '\n')
    print("第%s轮局部搜索次数为： %s" % (run_id, Nls))
    resultPop = tempTotalPop  # 这么多代后最终的族群

    # 对得到的最佳种子进行解码出A矩阵(即学生掌握知识点情况矩阵)和S，G向量()
    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)  # func79行
            A = acquireA(resultPopx, n_students, n_knowledge)  # 学生*知识点矩阵
            # Write_A(A, data_patch_id, run_id)
            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)  # 学生*知识点矩阵
            return A

    # 输出最优解
    index = np.argmax([ind.fitness for ind in resultPop])  # 返回的是最大适应度的索引
    if flag_train:
        slip, guess = decode(resultPop[index])

    # Write_sg(slip, guess, data_patch_id, run_id)
    # bestA = acquireA(resultPop[index], n_students, n_knowledge)
    # bestYITA = acquireYITA(bestA, q_matrix, n_students, n_questions, n_knowledge)
    # bestX, Xscore = acquireX(n_students, n_questions, bestYITA, slip, guess)
    # Write_X(bestX, data_patch_id, run_id)
    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]
    Accuracy = fit_max / (n_students * n_questions)  # 准确率计算
    print(str(gen) + '  Accuracy=' + str(Accuracy) + ' 最优适应度：' + str(tempTotalPop[index].fitness))
    # gen 为代数，
    # now = time.time()
    # local_time = time.localtime(now)
    # date_format_localtime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    # print(date_format_localtime)

    # 训练结果可视化：最大值
    gen = logbook.select('gen')  # 用select方法从logbook中提取迭代次数
    fit_maxs = logbook.select('max')  # 提取适应度最大值
    # print("fit_max: ",fit_maxs)
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 整个网格只有一个（1*1）
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')  # x轴代数 y轴适应度最大值
    ax.legend(loc='best')  # 显示图中的标签
    ax.set_xlabel('Generation')  # x轴坐标
    ax.set_ylabel('Fitness')  # y轴坐标
    fig.tight_layout()  # 自动调整子图参数

    if not flag_train:
        path_pre = 'czy_results/pict/test/MaxFitness_perGen_alg'
    else:
        path_pre = 'czy_results/pict/train/MaxFitness_perGen_alg'
    os.makedirs(path_pre, exist_ok=True)  # 递归目录创建函数
    # alg_name='GA_NBC', data_name
    fig.savefig(path_pre + alg_name + '_data：' + data_name + ' 共' + str(max_generations) + '代：   第' + str(
        data_patch_id) + '折' + '第' + str(run_id) + '次训练' + '.png')
    return resultPop, logbook, slip, guess


def GA(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess, data_patch_id,
       run_id, n_pop,
       flag_train, max_generations, len_s_g):
    n_knowledge = n_knowledge_fine
    # 优化目标：单变量，求最大值
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # 创建Individual类，继承list，求最大值
    creator.create('Individual', list, fitness=creator.FitnessMax)
    # 基因长度
    if flag_train:
        GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g  # 32*11+15*2*7
    else:
        GENE_LENGTH = n_students * n_knowledge  # 学生数8*知识点数11

    toolbox = base.Toolbox()
    # 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
    toolbox.register('Binary', bernoulli.rvs, 0.5)  # 等概率取0和1
    # 注册用tools.initRepeat生成长度为GENE_LENGTH的Individual、等概率取0和1、个体的基因长度
    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)
    # 注册生成群落的函数
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    # 生成初始族群
    pop = toolbox.Population(n=n_pop)

    # 定义评价函数
    def evaluate(individual):
        # 从生成的种子中提取随机生成的学生*知识点掌握情况矩阵A
        A = acquireA(individual, n_students, n_knowledge)

        # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量
        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
        slip_ratio = slip
        guess_ratio = guess
        if flag_train:
            # 此处的S和G是数字形式，不是0101形式
            s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)  # 列表 大小为question
            slip_ratio = s
            guess_ratio = g

        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)

        # 计算评价指标
        # data:学生答题 固定的文件、X:考虑SG的学生答题情况矩阵
        sum_ = 0
        for i in range(n_students):
            for j in range(n_questions):
                sum_ = sum_ + (1 - abs(data[i][j] - X[i][j]))  # 符合则sum+1 不符则+0

        return (sum_),  # 返回的sum越大说明data与X越匹配

    # 在工具箱中注册遗传算法需要的工具
    toolbox.register('evaluate', evaluate)
    # 注册Tournsize为2的锦标赛选择
    toolbox.register('select', tools.selTournament, tournsize=2)
    # 交叉算子：cxUniform为均匀交叉，交叉概率为0.5
    toolbox.register('mate', tools.cxUniform, indpb=0.6)  # 注意这里的indpb需要显示给出
    # 变异算子：mutFlipBit 为位翻转突变：对个体中的每一个基因按给定对变异概率取非。概率位0.3
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.003)

    # 注册计算过程中需要记录的数据 也是数据记录
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # 计算适应度的函数
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 注册记录
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields  # 表头gen nevals avg std min max

    # 评价族群
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]  # 无效价值的个体列表
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)  # 计算他们各自的适应度 适应度列表
    # for ind in pop:
    #     ind = local_search_knowledge(data, ind, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH,
    #                                  len_s_g)
    #     ind.fitness.values = evaluate(ind)
    #     print(ind.fitness.values)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit  # 赋值适应度
        # print(fit)

    # 根据族群适应度，编译出stats记录
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)  # 第0代
    tempTotalPop = []

    # 下面开始迭代！！！
    for gen in range(1, max_generations + 1):  # 训练代数全局变量 方便输出数据好看
        print('当前第' + str(gen) + "代")
        if gen % 50 == 0:
            print('当前第' + str(gen) + "代")
        if (gen == 1):
            totalPop = invalid_ind
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)  # 汉明距离二维列表
        # print("开始NBC算法"+str(computeTime()))
        # newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)  # NBC算法后
        # print("newg中个体数量: ", len(newg))
        # 对所有类当中的每一类进行循环交叉和突变

        subPop = totalPop

        cxpb = 0.8215
        # mutpb = (0.9 - ((0.9 - 0.3) / (max_generations)) * gen)  # 变异率
        mutpb = 0.004448

        N_subPop = len(subPop)
        ind_seed = subPop[0]
        # oldfitness = evaluate(ind)
        # seed_LS = toolbox.clone(ind_seed)
        # if flag_train:
        #     if n_knowledge_coarse == n_knowledge_fine:
        #         seed_LS = local_search_train_0(data, seed_LS, q_matrix, n_students, n_knowledge_fine, n_questions,
        #                                        GENE_LENGTH, len_s_g)
        #     else:
        #         seed_LS = local_search_train(data, seed_LS, q_matrix, n_students, n_knowledge_coarse, n_knowledge_fine,
        #                                      n_questions, GENE_LENGTH, len_s_g)
        # else:
        #     if n_knowledge_coarse < n_knowledge_fine:
        #         seed_LS = local_search_test(data, seed_LS, q_matrix, n_students, n_questions, n_knowledge_coarse,
        #                                     n_knowledge_fine, slip, guess)
        #     else:
        #         seed_LS = local_search_test_0(data, seed_LS, q_matrix, n_students, n_knowledge, slip, guess)
        # seed_LS.fitness.values = evaluate(seed_LS)
        # newfitness = evaluate(newind)

        # 配种选择
        seed_LS = ind_seed
        offspring = toolbox.select(subPop, N_subPop)
        # 一定要复制，否则在交叉和突变这样的原位操作中，会改变所有select出来的同个体副本
        offspring_Xor = []  # [toolbox.clone(ind) for ind in offspring]
        for i in range(N_subPop):
            offspring_Xor.append(copy.deepcopy(offspring[i]))
            offspring_Xor.append(copy.deepcopy(seed_LS))

        # 变异操作 - 交叉     所有后代中相邻两个个体一定几率交叉
        for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 评价当前没有fitness的个体，确保整个族群中的个体都有对应的适应度
        invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 变异操作 - 突变
        offspring_mut = [toolbox.clone(ind) for ind in offspring]
        for mutant in offspring_mut:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 评价当前没有fitness的个体，确保整个族群中的个体都有对应的适应度
        invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        offspring = offspring_Xor + offspring_mut + subPop
        offspring.append(seed_LS)
        # TODO: 选择一部分精英，再随机化一部分个体

        # 环境选择 - 保留精英
        # 对整个子种群进行排序
        offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')  # 选择精英,保持种群规模len(offspring)
        # pop_selected = Queue.Queue()
        pop_selected = []
        pop_selected.append(offspring[0])
        num_selected = 1
        # pop_selected.pop()
        for i in range(1, len(offspring)):
            idx = pop_selected[num_selected - 1]
            dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
            if dis >= 5:
                pop_selected.append(offspring[i])
                num_selected = num_selected + 1
        pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')
        # 剔除冗余个体

        # idx = 0
        # for i in range(1, len(offspring)):
        # offspring(idx,:)
        # cmp(offspring(idx,:), offspring(i,:))

        tempTotalPop.extend(pop_selected)

        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)  # 每一代记录一下

    resultPop = tempTotalPop  # 这么多代后最终的族群

    # 对得到的最佳种子进行解码出A矩阵(即学生掌握知识点情况矩阵)和S，G向量()
    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)
            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)  # 学生*知识点矩阵
            return A

    # 输出最优解
    index = np.argmax([ind.fitness for ind in resultPop])
    if flag_train:
        slip, guess = decode(resultPop[index])

    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]
    Accuracy = fit_max / (n_students * n_questions)  # 准确率计算
    print(str(gen) + ' ' + str(Accuracy) + ' 最优适应度：' + str(tempTotalPop[index].fitness))

    # now = time.time()
    # local_time = time.localtime(now)
    # date_format_localtime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    # print(date_format_localtime)

    # 训练结果可视化：最大值
    gen = logbook.select('gen')  # 用select方法从logbook中提取迭代次数
    fit_maxs = logbook.select('max')  # 提取适应度最大值
    # print("fit_max: ",fit_maxs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')  # 代数 适应度最大值
    ax.legend(loc='best')
    ax.set_xlabel('Generation Iteration')
    ax.set_ylabel('Fitness')

    fig.tight_layout()

    if not flag_train:
        path_pre = 'results/pict/test/Generation_Fitness_Max'
    else:
        path_pre = 'results/pict/train/Generation_Fitness_Max'
    os.makedirs(path_pre, exist_ok=True)
    fig.savefig(path_pre + str(max_generations) + '第' + str(data_patch_id) + '折' + '第' + str(run_id) + '次训练' + '.png')
    return resultPop, logbook, slip, guess


# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_train_0(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    slip, guess = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
    #
    # # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量
    # YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
    # slip = [0.00195, 0.00011, 0, 0, 0, 0.00094, 0, 0.15, 0, 0.0135, 0.02021, 0.00034, 0.03952, 0.15, 0.11941, 0.15]
    # guess = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00004, 0.05, 0.05]
    # slip = [0.00195, 0.00011, 0, 0, 0, 0.00094, 0, 0.4414, 0, 0.0135, 0.02021, 0.00034, 0.03952, 0.63529, 0.11941,
    #   0.23949]
    # guess = [0.84899, 0.66774, 0.50633, 0.73786, 0.31454, 0.61851, 0.69462, 0.09244, 0.69973, 0.55275, 0.49085, 0.50671,
    #  0.32233, 0.00004, 0.17133, 0.06438]
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, n_knowledge, n_students)
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对
    slip, guess = MStep(IL, r_matrix, data, n_knowledge, n_students, n_questions)
    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行
    # 个体中更新学生的知识掌握情况
    # TODO：并行可能会出错
    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)  # 667行 每个个体更新了染色体
    return individual  # 返回更新后的个体


# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_train(data, individual, q_matrix, n_students, n_knowledge_coarse, n_knowledge_fine, n_questions,
                       GENE_LENGTH, len_s_g, data_name):
    slip, guess = acquireSandG(n_students, individual, n_knowledge_fine, GENE_LENGTH, len_s_g)
    #
    # # 获取不考虑S和G的情况下，学生的答题情况矩阵YITA:其行数为学生数，列数为题目数量
    # YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
    # TODO: 应单独拧出来，允许反复创建
    if data_name != 'Math_DMiC':
        groups, groups_2 = random_group(n_knowledge_coarse, n_knowledge_fine)  # 609行 返回粗粒度与细粒度的对应关系
    else:
        groups, groups_2 = read_knowledge_group_from_data(n_knowledge_coarse,
                                                          n_knowledge_fine)  # 577行 读取Math_DMiC 问题的粒度
    # TODO: q_matrix应根据groups进行处理（合并）
    q_coarse = covert_q_matrix_coarse(q_matrix, groups, n_knowledge_coarse, n_questions)  # 564行  返回问题的粒度
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_coarse, n_knowledge_coarse, n_students)
    slip, guess = MStep(IL, r_matrix, data, n_knowledge_coarse, n_students, n_questions)

    individual = updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups)
    # if not math.isnan(slip[0]) or not math.isnan(guess[0]):  h这里会出现 slip[0]为nan然后进入的情况造成异常
    if not math.isnan(slip[0]) and not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge_fine, GENE_LENGTH, len_s_g)
    return individual


def covert_q_matrix_coarse(q_matrix, groups, n_knowledge_coarse, n_questions):
    q_coarse = np.zeros((n_questions, n_knowledge_coarse))
    for i in range(n_questions):
        for j in range(n_knowledge_coarse):
            group = groups[j]
            if sum([q_matrix[i][val] for val in group]) >= 1:
                q_coarse[i][j] = 1
            else:
                q_coarse[i][j] = 0

    return q_coarse  # 返回问题粒度


def read_knowledge_group_from_data(n_knowledge_coarse, n_knowledge_fine):
    # 读取 Math_DMiC 集中
    # if granularity >= 3 or granularity < 0:
    #     print("Error!")
    if n_knowledge_coarse == n_knowledge_fine:  # 知识点数
        granularity = 0
    elif n_knowledge_fine == 27:
        granularity = 1
    elif n_knowledge_fine == 170:
        granularity = 2
    else:
        print("Error!")
    file_path = "dataSets/Math_DMiC/q/q_groups_"  # .csv"
    file_path = file_path + str(granularity) + ".csv"
    groups = {}
    groups_2 = {}

    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        data = []
        temp = []
        num_line = 0
        for line in reader:
            line_split = line[0].split(";")  # ;为分隔符
            line_split = [int(each) for each in line_split]
            for each in line_split:
                groups.setdefault(num_line, []).append(each - 1)  # 查找值为num_line
                groups_2[each] = num_line
            temp.append(line_split)
            num_line = num_line + 1
    return groups, groups_2


def random_group(n_knowledge_coarse, n_knowledge_fine):
    if n_knowledge_coarse > n_knowledge_fine:
        print('Error: n_knowledge_coarse > n_knowledge_fine')
    groups = {}
    groups_2 = {}

    size_group = math.ceil(n_knowledge_fine / n_knowledge_coarse)  # cell函数，返回数值上的整数
    # 在知识层次未知的情况下，将细粒度的知识点随机整合为一个“粗”粒度的知识点
    num_selected = np.zeros(n_knowledge_coarse)
    max_num_selected = np.zeros(n_knowledge_coarse)
    for i in range(n_knowledge_coarse - 1):
        max_num_selected[i] = size_group
    max_last_idx = n_knowledge_fine - size_group * (n_knowledge_coarse - 1)
    max_num_selected[n_knowledge_coarse - 1] = max_last_idx
    for i in range(n_knowledge_fine):
        id_rnd = random.randint(1, n_knowledge_coarse) - 1  # 生成在1~n_knowledge_coarse随机一个数
        while num_selected[id_rnd] >= max_num_selected[id_rnd]:
            id_rnd = (id_rnd + 1) % n_knowledge_coarse
        num_selected[id_rnd] = num_selected[id_rnd] + 1
        groups.setdefault(id_rnd, []).append(i)
        groups_2[i] = id_rnd

    return groups, groups_2  # 返回粗粒度与细粒度的对应关系


# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_test_0(data, individual, q_matrix, n_students, n_knowledge, slip, guess):
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, n_knowledge, n_students)

    individual = updateIndividual_A(individual, A, n_students, n_knowledge)
    return individual


# 针对
# 根据个体individual中的s和g，更新学生的知识掌握情况
def local_search_test(data, individual, q_matrix, n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, slip,
                      guess, data_name):
    # TODO: 应单独拧出来，允许反复创建
    if data_name != 'math_DMiC':
        groups, groups_2 = random_group(n_knowledge_coarse, n_knowledge_fine)
    else:
        groups, groups_2 = read_knowledge_group_from_data(n_knowledge_coarse, n_knowledge_fine)
    # TODO: q_matrix应根据groups进行处理（合并）
    q_coarse = covert_q_matrix_coarse(q_matrix, groups, n_knowledge_coarse, n_questions)
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_coarse, n_knowledge_coarse, n_students)
    individual = updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups)

    return individual


def updateIndividual_A_0(individual, A, n_students, n_knowledge):
    individual[0:n_students * n_knowledge] = A  # GENE_LENGTH = student * knowlege + question * 2 * len_s_g
    return individual


def updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups):
    # individual[0:n_students * n_knowledge] = A  # GENE_LENGTH = student * knowlege + question * 2 * len_s_g
    # TODO: test
    for j in range(n_students):
        for i in range(n_knowledge_coarse):
            for idx in groups[i]:
                individual[j * n_knowledge_fine + idx] = A[j * n_knowledge_coarse + i]
    return individual


def updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH, len_s_g):
    # 共question个i, GENE_LENGTH = student * knowledge + question * 2 * len_s_g
    loop = 0
    for i in range(n_knowledge * n_students, GENE_LENGTH, len_s_g * 2):
        # TODO: 是否可以采用混合编码的方式？
        # 这里可能slip中也含有nan
        if not math.isnan(slip[loop]) and not math.isnan(guess[loop]):  # 检查s，g为数值，decode 解码过程
            individual[i:i + len_s_g] = decode_slip(slip[loop], len_s_g)  # 从n_knowledge开始，s为len_s_g的精度，调用 func的231行
            individual[i + len_s_g:i + (len_s_g * 2)] = decode_guess(guess[loop], len_s_g)
        loop = loop + 1

    return individual  # 每个个体的染色体已完成计算。


'''
输入：
IL:学生数*知识点模式数目（2*k)、
sg：分为为失误率（slip）、猜测率（guess）的当前值
n：学生-试题答题情况(Xij)
Q:试题-知识点关联情况
r：r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
k：知识点数目
输出：计算IL
'''


def EStep(slip, guess, data, q_matrix, n_knowledge, n_students):
    slip = np.array(slip)
    guess = np.array(guess)
    data = np.array(data)
    q_matrix = np.mat(q_matrix)  # 知识考察矩阵
    # crate K matrix，indict k skill could get how many vector
    # 构造K矩阵，表示k个技能可以组成的技能模式矩阵
    k_matrix = np.mat(np.zeros((n_knowledge, 2 ** n_knowledge), dtype=int))  # k行 8行 、 2**k列 256列
    for j in range(2 ** n_knowledge):  # 0到255 共256个数
        l = list(bin(j).replace('0b', ''))  # 范围是0到255的二进制 0000 0000-1111 1111 ，l为256种模式
        for i in range(len(l)):  # 0到7 l的长度为8
            k_matrix[n_knowledge - len(l) + i, j] = l[i]  # 8行 256列 的所有可能的技能矩阵赋值完毕
    '''8行 256列
    0 0......1
    0 0......1
    0 0......1
    0 0......1

    0 0......1
    0 0......1
    0 0......1
    0 1......1
    '''

    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    aPrior = np.ones(2 ** n_knowledge) / 10 ** 8
    std = np.sum(q_matrix, axis=1)  # 将每一行的元素相加,将Q矩阵压缩为一列 20行1列,每个题目有多少个知识点的阵
    '''
    # rer= Q * K 20*256 
    第一行 0 0 1 1 1 1 2 2 0 0 - 1 1 1 1 2 2 1 1 2 2 - 2 2 3 3 1 1 2 2 2 2 - ...前30个共256
    # rerb = (Q * K == std)  # 20*256 false or True
    第一行 
    False False False False False False False False False False 
    False False False False False False False False False False 
    False False  True  True False False False False False False...前30个共256
    '''
    r_matrix = (q_matrix * k_matrix == std) * 1  # Q:20x8 K:8x256  r:20x256  q_matrix为知识考察矩阵，K为8个知识点技能模式矩阵
    # r后面保持不变 ，  就是，考察，在学生的每个题目的知识点技能矩阵下，是否满足这道题所考察的知识点，如果满足，则r_matrix为1
    # r矩阵（r:20x256）表示理论上j这道题目对于l这个模式能否做对
    IL = np.zeros((n_students, 2 ** n_knowledge))  # 行：学生数，列为2^8
    for s in range(0,3):
        for l in range(2 ** n_knowledge):  # 若k=8: 256列 for循环256次 赋值256次
        # 学生的数量
            lll = ((1 - slip) ** data * slip ** (1 - data)) ** r_matrix.T.A[l] * (guess ** data * (
                1 - guess) ** (1 - data)) ** (1 - r_matrix.T.A[l])  # Xi的边缘似然函数L(Xi|αi)
            IL[:, l] = lll.prod(axis=1)  # prod连乘函数，L（X|α)，当有I个学生时的全体学生边缘化似然函数
        sumIL = IL.sum(axis=1)  # 一行元素相加 428行*1列
    # LX = np.sum([i for i in map(math.log2, sumIL)])  # 似然函数
    # print(LX)
        IL = (IL.T / sumIL).T* aPrior

    # E-step：现在得到了IL 428行学生*256列知识点模式
    A = []
    for i in range(n_students):
        idx = IL[i].argmax()  # 每个学生的最大似然函数
        tmp = k_matrix[:, idx].data.tolist()  # k矩阵 8*256 每个题的25
        tmp_array = [i[0] for i in tmp]
        A = A + tmp_array

    return A, IL, k_matrix, r_matrix
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对


'''
输入：IL,n,r,k,i
输出：IR
'''


def MStep(IL, r_matrix, data, n_knowledge, n_students, n_questions):
    data = np.array(data)
    # A为求得的能力属性掌握矩阵I*K , IL全体学生边缘似然函数，K_matrix为8行*256列的所有可能的技能矩阵，r_matrix为表示理论上j这道题目对于l这个模式能否做对
    # IR中的 0 1 2 3  分别表示 IO RO I1 R1
    IR = np.zeros((4, n_questions))  # 4行20列
    n1 = np.ones((n_students, n_questions))  # 428*20 全1矩阵
    for l in range(2 ** n_knowledge):  # 256次循环
        IR[0] += np.sum(((1 - r_matrix.A[:, l]) * n1).T * IL[:, l], axis=1)  # I0，至少缺乏习题j关联的一个知识点的期望学生数
        IR[1] += np.sum(((1 - r_matrix.A[:, l]) * data).T * IL[:, l], axis=1)  # R0，IO中正确答对第J道题的期望学生数目
        IR[2] += np.sum((r_matrix.A[:, l] * n1).T * IL[:, l], axis=1)  # I1 ，掌握了第j道题所需所有知识点的期望学生数目
        IR[3] += np.sum((r_matrix.A[:, l] * data).T * IL[:, l], axis=1)  # R1 ，I1中正确作答题目J的期望学生数目
    # 针对每一道题目，根据I0,R0,I1,R1，来更新s和g，更新后的sg，又重新计算似然函数矩阵IL
    # if (abs(IR[1] / IR[0] - sg[:,1])<threshold).any() and (abs((IR[2]-IR[3]) / IR[2] -sg[:,0])<threshold).any():

    guess = IR[1] / IR[0]  # 更新g猜测率
    slip = (IR[2] - IR[3]) / IR[2]  # 更新s失误率
    # if math.isnan(slip[0]):
    #     print(str(IR[0]))
    for i in range(n_questions):
        if slip[i] > threshold_slip:
            slip[i] = threshold_slip
        if guess[i] > threshold_guess:
            guess[i] = threshold_guess
    # M-step结束 得到IR 0123

    return slip, guess
