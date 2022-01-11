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
warnings.filterwarnings("ignore")


def single_run(data_patch_id, run_id, num_train_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g,
               data_train, q_matrix, data_test, num_test_students, alg_name, data_name, max_generations, n_pop):
    random.seed(run_id)
    f_record, f_record_data = files_open(alg_name, data_name, data_patch_id, run_id)
    print("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次训练")

    slip, guess = trainModel(num_train_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data_train,
                             q_matrix, data_patch_id, run_id, n_pop, max_generations, alg_name, data_name)

    print("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次测试")
    accuracy, precision, recall, f1, auc = testModel(num_test_students, n_questions, n_knowledge_coarse,
                                                     n_knowledge_fine, len_s_g, data_test,
                                                     q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                     max_generations, alg_name, data_name)  # testModel.py  6行
    f_record.writelines("第" + str(data_patch_id) + "折" + "第" + str(run_id) + "次" + "\n" + "Accuracy:" + str(accuracy) +
                        "\t" + "Precision:" + str(precision) + "\t" + "Recall:" + str(recall) + "\t" + "F1:" + str(f1) +
                        "\t" + "AUC:" + str(auc) + "\t" + "\n")

    f_record_data.writelines(str(data_patch_id) + "." + str(run_id) + "\t" + str(accuracy) + "\t" +
                             str(precision) + "\t" + str(recall) + "\t" + str(f1) + "\t" + str(auc) +
                             "\t" + "\n")
    files_close(f_record, f_record_data)
    return accuracy, precision, recall, f1, auc, run_id


def prepare(dataset_idx=0, granularity=0):

    data_name = data_names[dataset_idx]

    if data_name == 'Math_DMiC':
        q_matrix_0, q_matrix_1, q_matrix_2, data = data_reader_math_DMiC(data_name)

        if granularity == 0:
            q_matrix = q_matrix_0
        elif granularity == 1:
            q_matrix = q_matrix_1
        elif granularity == 2:
            q_matrix = q_matrix_2
        q_matrix_coarse = q_matrix_0
    else:
        q_matrix, data = data_reader(data_name)
        q_matrix_coarse = q_matrix
    return data, q_matrix, q_matrix_coarse


def run_main(alg_name, data_name, data, q_matrix, max_runs, max_split, max_generations, n_pop, multi,
             n_knowledge_coarse):

    KF = KFold(n_splits=max_split, shuffle=False)
    data_patch_i = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    for train_index, test_index in KF.split(data):
        data_patch_i = data_patch_i + 1
        data_train, data_test = data[train_index], data[test_index]

        num_train_students = len(data_train)
        num_test_students = len(data_test)
        print('训练集学生数' + str(num_train_students))
        print('测试集学生数' + str(num_test_students))

        n_questions = len(q_matrix)
        n_knowledge = len(q_matrix[0])
        len_s_g = 7

        n_knowledge_fine = n_knowledge

        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_f1 = 0
        mean_auc = 0

        array_accuracy = []
        array_precision = []
        array_recall = []
        array_f1 = []
        array_auc = []
        results_per_run = []

        if multi:
            print('multi：' + str(max_runs) + ' processes')
            pool = Pool(processes=max_runs)

            multiple_results = []
            for run_id in range(max_runs):

                multiple_results.append(
                    pool.apply_async(single_run, (data_patch_i, run_id, num_train_students, n_questions,
                                                  n_knowledge_coarse, n_knowledge_fine, len_s_g, data_train,
                                                  q_matrix, data_test, num_test_students, alg_name,
                                                  data_name, max_generations, n_pop)))

            pool.close()
            pool.join()

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
            for run_id in range(max_runs):

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
    is_multi = True
    max_generations = 50
    n_pop = 100

    alg_names = ["HGA_CDM_IALS","HGA_CDM_ALS","HGA_CDM_LS","GA_CDM"]
    data_names = ["FrcSub",  "Math2", "Math1"]
    alg_id = 0
    dataset_id = 0
    n_pop = 100
    granularity = 1
    NumTime = 0
    sum_ALS = 0
    for alg_id in range(3,4):
        for dataset_id in range(0, 1):

            NumTime = NumTime + 1
            alg_name = alg_names[alg_id]
            data_name = data_names[dataset_id]
            print_info = str(NumTime) + "##" + str(alg_id) + " " + alg_name + " " + str(
                dataset_id) + " " + data_name
            print(print_info)
            data, q_matrix, q_matrix_coarse = prepare(dataset_id, granularity)

            n_questions, n_knowledge_coarse = np.mat(q_matrix).shape

            run_main(alg_name, data_name, data, q_matrix, max_runs, max_split, max_generations, n_pop, is_multi,
                     n_knowledge_coarse)



    print('总用时：' + str(time.time() - startTimeA) + '秒')
