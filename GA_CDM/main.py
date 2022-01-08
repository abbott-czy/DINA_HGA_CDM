from train_model import *
from testModel import *
from multiprocessing import Pool
from sklearn.model_selection import KFold
import warnings
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', 1000)
warnings.filterwarnings("ignore")  # 忽略正常运行时的错误


def single_run(data_patch_id, run_id, num_train_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g,
               data_train, q_matrix, data_test, num_test_students, alg_name, data_name, max_generations, n_pop):
    random.seed(run_id)  # 固定每次运行的随机种子为run_id# ，以复现实验结果
    f_record, f_record_data = files_open(alg_name, data_name, data_patch_id, run_id)  # tools.py 106行
    print("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次训练")
    # 参数：训练人数、题目数、知识点数、sg精度、学生*题目二维列表、试题*知识点二维列表、训练次数序号
    slip, guess = trainModel(num_train_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data_train,
                             q_matrix, data_patch_id, run_id, n_pop, max_generations, alg_name, data_name)

    # guess_record.append(guess)
    # slip_record.append(slip)

    # 测试
    print("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次测试")
    # 参数：测试人数、题目数、知识点数、sg精度、学生*题目二维列表、试题*知识点二维列表、sg、训练次数序号
    accuracy, precision, recall, f1, auc = testModel(num_test_students, n_questions, n_knowledge_coarse,
                                                     n_knowledge_fine, len_s_g, data_test,
                                                     q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                     max_generations, alg_name, data_name)  # testModel.py  6行
    f_record.writelines("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次" + "\n" + "Accuracy:" + str(accuracy) +
                        "\t" + "Precision:" + str(precision) + "\t" + "Recall:" + str(recall) + "\t" + "F1:" + str(f1) +
                        "\t" + "AUC:" + str(auc) + "\t" + "\n")
    # 不添加第几折，第几次训练
    f_record_data.writelines(str(data_patch_id) + "." + str(run_id) + "\t" + str(accuracy) + "\t" +
                             str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\t" + str(auc) +
                             "\t" + "\n")
    files_close(f_record, f_record_data)
    return accuracy, precision, recall, f1, auc, run_id


def prepare(dataset_idx=0, granularity=0):
    # 数据读取初始化
    data_name = data_names[dataset_idx]

    if data_name == 'Math_DMiC':
        q_matrix_0, q_matrix_1, q_matrix_2, data = data_reader_math_DMiC(data_name)
        # data_reader_math_DMiC 位于tools.py文件
        if granularity == 0:
            q_matrix = q_matrix_0
        elif granularity == 1:
            q_matrix = q_matrix_1
        elif granularity == 2:
            q_matrix = q_matrix_2
        q_matrix_coarse = q_matrix_0  # 粗粒度
    else:
        q_matrix, data = data_reader(data_name)
        q_matrix_coarse = q_matrix
    return data, q_matrix, q_matrix_coarse


def run_main(alg_name, data_name, data, q_matrix, max_runs, max_split, max_generations, n_pop, multi,
             n_knowledge_coarse):
    # 传入参数：alg_name：算法， data_name：数据集，学生作答矩阵data，知识考察矩阵q_matrix，max_runs：多进程参数
    # max_split 分折数， max_generations=遗传代数 ， n_pop ：种群人数 , is_multi：是否进行多进程
    # n_knowledge_coarse 知识点粒度数
    # i为第几次训练和测试 一共五次
    # KF = KFold(n_splits=max_split, shuffle=False, random_state=0)  # 洗牌打乱
    KF = KFold(n_splits=max_split, shuffle=False)
    data_patch_i = 0  # i为第几次训练和测试 总次数为max_split
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    for train_index, test_index in KF.split(data):  # 这里将data划分5次h
        data_patch_i = data_patch_i + 1
        data_train, data_test = data[train_index], data[test_index]

        num_train_students = len(data_train)  # 32/120
        num_test_students = len(data_test)  # 8/120
        print('训练集学生数' + str(num_train_students))
        print('测试集学生数' + str(num_test_students))

        n_questions = len(q_matrix)  # 题目数
        n_knowledge = len(q_matrix[0])  # 知识点数
        len_s_g = 7  # s,g的精度

        n_knowledge_fine = n_knowledge

        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_f1 = 0
        mean_auc = 0

        array_accuracy = []  # 准确率   正例： TP真正例：答对的1  FP假正例 ：猜测对的
        array_precision = []  # 精确率，查准率        反例： FN假反例：失误而打错的   TN真反例：打错为0
        array_recall = []  # 召回率，查全率
        array_f1 = []  # 查准率和查全率的一个加权平均
        array_auc = []  # auc就是曲线下面积，这个数值越高，则分类器越优秀
        results_per_run = []

        if multi:
            print('multi：' + str(max_runs) + ' processes')
            pool = Pool(processes=max_runs)  # 进程个数为max_runs
            # 训练
            multiple_results = []  # 训练结果
            for run_id in range(max_runs):
                # single_run返回：accuracy, precision, recall, f1, auc, run_id
                multiple_results.append(
                    pool.apply_async(single_run, (data_patch_i, run_id, num_train_students, n_questions,
                                                  n_knowledge_coarse, n_knowledge_fine, len_s_g, data_train,
                                                  q_matrix, data_test, num_test_students, alg_name,
                                                  data_name, max_generations, n_pop)))
                # pool.apply_async:异步非阻塞,主进程和子进程同时跑，谁跑的快，谁先来。
                # data_patch_i :第几折  ，run_id 第几次训练 ，n_questions：题目数
                # len_s_g s,g的精度 ，n_knowledge_fine 知识点数
                # data_test测试集  data_train 数据训练集， num_test_students 测试集学生数，alg_name 算法选择
                # 传入参数：alg_name：算法， data_name：数据集，学生作答矩阵data，知识考察矩阵q_matrix，max_runs：多进程参数
                # max_split 分折数， max_generations=遗传代数 ， n_pop ：种群人数 , is_multi：是否进行多进程
                # n_knowledge_coarse 知识点粒度数
            pool.close()  # 关闭pool，使其不在接受新的（主进程）任务
            pool.join()  # 主进程阻塞后，让子进程继续运行完成，子进程运行完后，再把主进程全部关掉。

            for res in multiple_results:

                accuracy, precision, recall, f1, auc, run_id = res.get()

                mean_accuracy += accuracy
                mean_precision += precision
                mean_recall += recall
                mean_f1 += f1
                mean_auc += auc
                array_precision.append(precision)
                array_recall.append(recall)
                array_f1.append(f1)
                array_auc.append(auc)
                results_per_run.append((run_id, accuracy, precision, recall, f1, auc))  # 每次的运行结果
        else:
            for run_id in range(max_runs):  # run_id更改随机种子，使得同一组数据训练次数为max_runs
                # 训练
                time.sleep(10)
                accuracy, precision, recall, f1, auc, run_id = single_run(data_patch_i, run_id, num_train_students,
                                                                          n_questions, n_knowledge_coarse,
                                                                          n_knowledge_fine, len_s_g, data,
                                                                          q_matrix, data, num_test_students,
                                                                          alg_name, data_name, max_generations, n_pop)
                mean_accuracy += accuracy
                mean_precision += precision
                mean_recall += recall
                mean_f1 += f1
                mean_auc += auc
                array_accuracy.append(accuracy)
                array_precision.append(precision)
                array_recall.append(recall)
                array_f1.append(f1)
                array_auc.append(auc)
                results_per_run.append((run_id, accuracy, precision, recall, f1, auc))
        mean_accuracy /= max_runs
        mean_precision /= max_runs
        mean_recall /= max_runs
        mean_f1 /= max_runs
        mean_auc /= max_runs
        save_final_results(results_per_run, data_patch_i, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc,
                           max_runs, alg_name, data_name)
        s1 += mean_accuracy
        s2 += mean_precision
        s3 += mean_recall
        s4 += mean_f1
        s5 += mean_auc
    average_accuracy = s1 / max_split
    average_precision = s2 / max_split
    average_recall = s3 / max_split
    average_f1 = s4 / max_split
    average_auc = s5 / max_split
    save_final_results_average(average_accuracy, average_precision, average_recall, average_f1, average_auc, max_runs,
                               alg_name, data_name)


if __name__ == '__main__':

    startTimeA = time.time()
    max_runs =20
    max_split = 5
    is_multi = True  # True  # True  # 是否采用多进程
    # 此处更改训练和测试的遗传代数
    max_generations = 50
    n_pop = 100
    # alg_names = ["GA_NBC_multi", "GA_NBC", "GA"]
    # data_names = ["FrcSub", "Math1", "Math2", "Math_DMiC"]  #这里稍微更改一下位置 h
    alg_names = ["GA_NBC_multi", "GA_NBC", "GA","HGA"]  # 在myALgorithms.py中
    data_names = ["FrcSub", "Math_DMiC", "Math2", "Math1"]
    alg_id = 0  # 多进程朴素贝叶斯分类算法
    dataset_id = 0
    n_pop = 100
    granularity = 1  # 针对math_DMiC数据集，0:8个知识点；1：27个知识点；2：170个知识点
    NumTime = 0  # 记录改变次数
    sum_ALS = 0
    for alg_id in range(3, 4):  # 这里把算法的第一个省略, 后面固定算法，改变粒度；这里固定粒度改变算法 range(1,2) 输出1
        for dataset_id in range(3, 4):

            NumTime = NumTime + 1
            alg_name = alg_names[alg_id]  # 使用的算法
            data_name = data_names[dataset_id]  # 使用的数据集
            print_info = str(NumTime) + "##" + str(alg_id) + " " + alg_name + " " + str(
                dataset_id) + " " + data_name
            print(print_info)  # 打印：记录改变次数，算法，和数据集
            data, q_matrix, q_matrix_coarse = prepare(dataset_id, granularity)
            if data_name != 'Math_DMiC' and alg_name == 'GA_NBC_multi':
                n_knowledge_coarse = 4  # 参数 n_knowledge
            elif data_name == 'Math_DMiC' and alg_name == 'GA_NBC_multi':
                n_questions, n_knowledge_coarse = np.mat(q_matrix_coarse).shape  # j题目 与 k知识点粒度 矩阵
            else:  # data_name != 'Math_DMiC':
                n_questions, n_knowledge_coarse = np.mat(q_matrix).shape

            run_main(alg_name, data_name, data, q_matrix, max_runs, max_split, max_generations, n_pop, is_multi,
                     n_knowledge_coarse)

    with open('./czy_results/ALS/als.txt', 'r') as f:
        reader = f.readlines()
        for row in reader:
            # print(row)
            sum_ALS = sum_ALS + int(row)
    print("平均每轮ALS次数为 %d" % (sum_ALS / (max_split * max_runs)))
    # 传入参数：算法，数据集，学生作答矩阵data，知识考察矩阵q_matrix，max_runs：多进程参数
    # max_split 分折数， max_generations=遗传代数 ， n_pop ：种群人数 , is_multi：是否进行多进程
    # n_knowledge_coarse 知识点粒度数

    # alg_id = 0
    # NumTime = 0  # 记录改变次数
    # for granularity in range(3):
    #     for dataset_id in range(0, 1):
    #         NumTime = NumTime + 1
    #         alg_name = alg_names[alg_id]
    #         data_name = data_names[dataset_id]
    #         print_info =str(NumTime) + "##"+ str(alg_id) + alg_name + str(dataset_id) + data_name
    #         print(print_info)
    #         data, q_matrix, q_matrix_coarse = prepare(dataset_id, granularity)
    #         if data_name != 'Math_DMiC' and alg_name == 'GA_NBC_multi':
    #             n_knowledge_coarse = 4  # 参数 n_knowledge
    #         elif data_name == 'Math_DMiC' and alg_name == 'GA_NBC_multi':
    #             n_questions, n_knowledge_coarse = np.mat(q_matrix_coarse).shape
    #         elif data_name != 'Math_DMiC':
    #             n_questions, n_knowledge_coarse = np.mat(q_matrix).shape
    #
    #         run_main(alg_name, data_name, data, q_matrix, max_runs, max_split, max_generations, n_pop, is_multi,
    #                  n_knowledge_coarse)
    print('总用时：' + str(time.time() - startTimeA) + '秒')
