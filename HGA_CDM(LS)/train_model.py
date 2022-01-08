from myAlgorithms import *


def trainModel(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, data_patch_id, run_id=1, n_pop=50,
               max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):
    # 传入参数：alg_name：算法， data_name：数据集，学生作答矩阵data，知识考察矩阵q_matrix，max_runs：多进程参数
    # max_split 分折数， max_generations=遗传代数 ， n_pop ：种群人数 , is_multi：是否进行多进程，data为训练集人数。
    # n_knowledge_coarse 知识点粒度数
    # i为第几次训练和测试 一共五次
    flag_train = True
    slip = []
    guess = []

    # K_matrix = np.mat(np.zeros((n_knowlege, 2 ** n_knowlege), dtype=int))
    # for j in range(2 ** n_knowlege):
    #     l = list(bin(j).replace('0b', ''))
    #     for i in range(len(l)):
    #         K_matrix[n_knowlege - len(l) + i, j] = l[i]
    # std = np.sum(q_matrix, axis=1)
    # r_matrix = (q_matrix * K_matrix == std) * 1

    if alg_name =='GA':
        resultPop, logbook, slip, guess = GA(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data,
                                                 q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                 flag_train, max_generations, len_s_g)
    elif alg_name == 'GA_NBC' or alg_name == 'GA_NBC_multi' or alg_name == 'HGA' or alg_name == "HGA_LS":
        resultPop, logbook, slip, guess = GA_NBC(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data,
                                             q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                             flag_train, max_generations, len_s_g, alg_name, data_name)

    # now = time.time()
    # local_time = time.localtime(now)
    # date_format_localtime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    # print(date_format_localtime)

    return slip, guess

