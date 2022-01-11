import os
import random

import pandas as pd
from func import *


def saveC(array, path):
    data = pd.DataFrame(array)
    data.to_csv(path, index=False)


def data_reader(data_name):
    q_path = "dataSets/" + data_name + "/q.csv"

    data_path = "dataSets/" + data_name + "/data.csv"

    if os.path.exists(q_path) == False:
        q_path_pre = "dataSets/" + data_name + "/q"
        txt2csv(q_path_pre)

    if os.path.exists(data_path) == False:
        data_path_pre = "dataSets/" + data_name + "/data"
        txt2csv(data_path_pre)

    q_matrix = acquireQ(q_path)
    data = acquireData(data_path)

    data = np.array(data)

    return q_matrix, data


def data_reader_math_DMiC(data_name, ith=1):
    granularity = 0

    q_str = "q" + str(granularity)
    q_path_pre = "dataSets\\" + data_name + "\\q\\" + q_str + "\\" + q_str + "_" + str(ith)
    q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:
        txt2csv(q_path_pre)

    q_matrix_0 = acquireQ(q_path)

    granularity = 1

    q_str = "q" + str(granularity)
    q_path_pre = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + str(ith)
    q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:
        txt2csv(q_path_pre)

    q_matrix_1 = acquireQ(q_path)

    granularity = 2

    q_str = "q" + str(granularity)
    q_path_pre = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + str(ith)
    q_path = q_path_pre + ".csv"
    if os.path.exists(q_path) == False:
        txt2csv(q_path_pre)

    q_matrix_2 = acquireQ(q_path)


    data_path_pre = "dataSets/" + data_name + "/data/data" + str(ith)
    data_path = data_path_pre + ".csv"
    if os.path.exists(data_path) == False:
        txt2csv(data_path_pre)
    data = acquireData(data_path)

    data = np.array(data)

    return q_matrix_0, q_matrix_1, q_matrix_2, data


def q_matrix_reader_math_DMiC(data_name, ith=1):
    granularity = 0

    q_str = "q" + granularity
    q_path = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + ith + ".csv"
    q_matrix_0 = acquireQ(q_path)

    granularity = 1
    q_str = "q" + granularity
    q_path = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + ith + ".csv"
    q_matrix_1 = acquireQ(q_path)

    granularity = 2

    q_str = "q" + granularity
    q_path = "dataSets/" + data_name + "/q/" + q_str + "/" + q_str + "_" + ith + ".csv"
    q_matrix_2 = acquireQ(q_path)

    return q_matrix_0, q_matrix_1, q_matrix_2


def files_open(alg, data_name, data_patch_i, runID):
    filename = "_results/Frusub_GA/" + alg + "_" + data_name + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    filename = "_results/Frusub_GA/" + alg + "_score_" + data_name + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record_data = open(filename, 'a')
    return f_record, f_record_data


def files_close(f_record, f_record_data):
    f_record.close()
    f_record_data.close()


def save_final_results(results_per_run, data_patch_id, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc,
                       max_runs, alg_name,
                       data_name):
    filename = "_results/Frusub_GA/alg_" + alg_name + "_data_" + data_name + "_" + str(data_patch_id) + "_max_run" + str(max_runs) + "_.txt"
    print(filename)
    str_results = str(mean_accuracy) + "    " + str(mean_precision) + "  " + str(mean_recall) + "   " + str(
        mean_f1) + "       " + str(mean_auc)
    str_index = " mean_accuracy       mean_precision       mean_recall      mean_f1         mean_auc  "
    print("accuracy, precision, recall, f1, auc  " + str_results)
    f_record = open(filename, 'a')
    f_record.writelines(str_index + '\n')
    f_record.writelines(str_results + '\n')
    str_index_1 = " run_id   mean_accuracy        mean_precision       mean_recall      mean_f1         mean_auc  "
    f_record.writelines(str_index_1 + '\n')
    for i in range(max_runs):

        f_record.writelines(str(results_per_run[i]) + '\n')

    f_record.close()


def save_final_results_average(average_accuracy, average_precision, average_recall, average_f1, average_auc, max_runs,
                               alg_name, data_name):
    filename = "_results/Frusub_GA/_average_alg_" + alg_name + "_data_" + data_name + "_max_run ：" + str(max_runs) + "_.txt"
    print(filename)
    str_average = str(average_accuracy) + "      " + str(average_precision) + "      " + str(
        average_recall) + "     " + str(average_f1) + "       " + str(average_auc)
    str_index = " mean_accuracy           mean_precision               mean_recall            mean_f1        " \
                "      mean_auc  "
    print("accuracy, precision, recall, f1, auc  " + str_average)
    f_record = open(filename, 'a')
    f_record.writelines(str_index + '\n')
    f_record.writelines(str_average + '\n')
    f_record.close()

def saveC(array, path):
    data = pd.DataFrame(array)
    data.to_csv(path, index=False)

def Write_sg(slip,guess,data_patch_i,runID):
    filename = "_results/reader/sg_" + "_" + str(data_patch_i) + "_th_" + str( runID) + ".txt"
    f_record = open(filename,'a')
    f_record.write("每个题的失误率slip：\n")
    for i in range(len(slip)):
        f_record.write(str(i)+": " + str(slip[i])+'   ')
    f_record.write("\n每个题的猜测率slip：\n")
    for j in range(len(guess)):
        f_record.write(str(j)+": "+str(guess[j])+'  ')
    f_record.close()



def Write_A(A,data_patch_i,runID):
    a = np.array(A)
    Ai, Aj = a.shape
    filename = "_results/reader/A_"+ "_" + str(data_patch_i) + "_th_" + str( runID) + ".txt"
    f_record = open(filename, 'a')
    for i in range(Ai):
        f_record.write("id"+str(i)+":  ")
        for j in range(Aj):
            f_record.write(str(A[i][j]) + '  ')
        f_record.write('\n')
    f_record.close()


def Write_X(X,data_patch_i,runID):
    x = np.array(X)
    xi, xj = x.shape
    filename = "_results/reader/X_"+ "_" + str(data_patch_i) + "_th_" + str( runID) + ".txt"
    f_record = open(filename, 'a')
    for i in range(xi):
        f_record.write("id" + str(i) + ":  ")
        for j in range(xj):
            f_record.write(str(X[i][j]) + '  ')
        f_record.write('\n')
    f_record.close()

def Write_rrrSum(rrrSum,data_patch_i,runID):
    Qi=300
    Qj=10
    filename = "_results/reader/rrrSum_"+ "_" + str(data_patch_i) + "_th_" + str( runID) + ".txt"
    f_record = open(filename, 'a')
    for i in range(Qi   ):
        f_record.write("id"+str(i)+":  ")
        for j in range(Qj):
            f_record.write(str(rrrSum[i*10+j]) + '  ')
        f_record.write('\n')
    f_record.close()
