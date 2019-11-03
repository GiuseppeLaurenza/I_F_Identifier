import os

import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score
from pycm import ConfusionMatrix
from statistics import stdev, mean
from sklearn.preprocessing import LabelBinarizer
from functools import reduce
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)


def compute_result_oc(folder):
    with open(folder + "oc_result_clean.p", "rb") as infile:
        data = pickle.load(infile)
    apt_detection_list = []
    apt_identification_full_list = []
    apt_identification_list = []
    for elem in data:
        current_pred = elem["pred"].replace(-1, 0)
        current_res = elem["res"].replace(-1, 0)
        apt_number = len(current_res.sum(1)[current_res.sum(1) == 1])
        dect_pred = current_pred.sum(1).values
        dect_pred[dect_pred > 1] = 1
        dect_true = current_res.sum(1).values
        cm = ConfusionMatrix(dect_true, dect_pred)
        cm_df = pd.DataFrame(cm.matrix).T.fillna(0)
        apt_detection_list.append(cm_df)

        current_pred_apt = current_pred[0:apt_number]
        current_res_apt = current_res[0:apt_number]
        current_pred_apt = current_pred_apt.replace(1, pd.Series(current_pred_apt.columns, current_pred_apt.columns))
        current_pred_apt["apt"] = current_pred_apt.apply(lambda x: [y for y in x if y != 0], 1)
        multilabel = current_pred_apt["apt"][current_pred_apt["apt"].apply(len) > 1].index
        # current_pred_apt["apt"] = current_pred_apt["apt"].apply(lambda x: x[0] if len(x)>0 else "Non APT",1)
        current_res_apt = current_res_apt.replace(1, pd.Series(current_res_apt.columns, current_res_apt.columns))
        current_res_apt["apt"] = current_res_apt.apply(lambda x: [y for y in x if y != 0][0], 1)
        y_pred = list(current_pred_apt["apt"])
        y_true = list(current_res_apt["apt"])
        for index in multilabel:
            for class_name in y_pred[index]:
                y_pred.append(class_name)
                y_true.append(y_true[index])
        not_found = [i for i, x in enumerate(y_pred) if len(x) == 0]
        found = [i for i, x in enumerate(y_pred) if i not in not_found]
        y_pred = [i for j, i in enumerate(y_pred) if (j not in multilabel and j not in not_found)]
        y_pred = [x[0] if (len(x) > 0 and isinstance(x, list)) else x for x in y_pred]
        y_true = [i for j, i in enumerate(y_true) if (j not in multilabel and j not in not_found)]
        cm = ConfusionMatrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm.matrix).T.fillna(0)
        diag = np.diag(cm_df)
        tp = sum(diag)
        total = sum(sum(cm_df.values))
        total_elements = current_res.iloc[found].sum().sort_index()
        tn = 0
        fp = 0
        fn = 0
        index = 0
        for current_elem in cm_df.iterrows():
            tn = tn + total - sum(current_elem[1])
            fp = fp + sum(current_elem[1]) - diag[index]
            fn = fn + total_elements[index] - diag[index]
            index = index + 1
        apt_identification_list.append(np.array([(tn, fn), (fp, tp)]))
        apt_identification_full_list.append(cm_df)
    detection_cm = sum(apt_detection_list)
    if not os.path.exists(folder+"oc/"):
        os.makedirs(folder+"oc/")
    detection_cm.to_csv(folder+"oc/apt_detection.csv",index=False)
    apt_identification_cm = pd.DataFrame(sum(apt_identification_list),columns=[0,1], index=[0,1])
    apt_identification_cm.to_csv(folder+"oc/apt_identification.csv", index=False)
    apt_identification_cm_full = sum(apt_identification_full_list)
    apt_identification_cm_full.to_csv(folder+"oc/apt_identification_full.csv", index=False)


def compute_result_rf(folder):
    with open(folder + "rf_result.p", "rb") as infile:
        loaded_data = pickle.load(infile)
    res_df = loaded_data["df_binarized"]
    for current_percentage in ["pred_5","pred_10","pred_15"]:
        data = loaded_data[current_percentage]
        apt_detection_list = []
        apt_identification_full_list = []
        apt_identification_list = []
        for elem in data:
            current_pred = elem.replace(-1, 0)
            apt_list = [x for x in current_pred.columns.values if x!="md5"]
            current_res = res_df[res_df["md5"].isin(current_pred["md5"])].replace(-1,0)
            apt_number = len(current_res[apt_list].sum(1)[current_res[apt_list].sum(1) == 1])
            dect_pred = current_pred.sum(1).values
            dect_pred[dect_pred > 1] = 1
            non_apt_number = len(dect_pred) - apt_number
            dect_true = [1] * apt_number + [0] * non_apt_number
            cm = ConfusionMatrix(dect_true, dect_pred)
            cm_df = pd.DataFrame(cm.matrix).T.fillna(0)
            apt_detection_list.append(cm_df)

            current_pred_apt = current_pred[0:apt_number].sort_values("md5")[apt_list]
            current_res_apt = current_res[0:apt_number].sort_values("md5")[apt_list]
            current_pred_apt = current_pred_apt.replace(1,
                                                        pd.Series(current_pred_apt.columns, current_pred_apt.columns))
            current_pred_apt["apt"] = current_pred_apt.apply(lambda x: [y for y in x if y != 0], 1)
            multilabel = current_pred_apt["apt"][current_pred_apt["apt"].apply(len) > 1].index
            # current_pred_apt["apt"] = current_pred_apt["apt"].apply(lambda x: x[0] if len(x)>0 else "Non APT",1)
            current_res_apt = current_res_apt.replace(1, pd.Series(current_res_apt.columns, current_res_apt.columns))
            current_res_apt["apt"] = current_res_apt.apply(lambda x: [y for y in x if y != 0][0], 1)
            y_pred = list(current_pred_apt["apt"])
            y_true = list(current_res_apt["apt"])
            for index in multilabel:
                for class_name in y_pred[index]:
                    y_pred.append(class_name)
                    y_true.append(y_true[index])
            not_found = [i for i, x in enumerate(y_pred) if len(x) == 0]
            found = [i for i, x in enumerate(y_pred) if i not in not_found]
            y_pred = [i for j, i in enumerate(y_pred) if (j not in multilabel and j not in not_found)]
            y_pred = [x[0] if (len(x) > 0 and isinstance(x, list)) else x for x in y_pred]
            y_true = [i for j, i in enumerate(y_true) if (j not in multilabel and j not in not_found)]
            cm = ConfusionMatrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm.matrix).T.fillna(0)
            diag = np.diag(cm_df)
            tp = sum(diag)
            total = sum(sum(cm_df.values))
            total_elements = current_res[apt_list].iloc[found].sum().sort_index()
            tn = 0
            fp = 0
            fn = 0
            index = 0
            for current_elem in cm_df.iterrows():
                tn = tn + total - sum(current_elem[1])
                fp = fp + sum(current_elem[1]) - diag[index]
                fn = fn + total_elements[index] - diag[index]
                index = index + 1
            apt_identification_list.append(np.array([(tn, fn), (fp, tp)]))
            apt_identification_full_list.append(cm_df)
        detection_cm = sum(apt_detection_list)
        path = folder + "rf/" + current_percentage + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        detection_cm.to_csv(path+"apt_detection.csv", index=False)
        apt_identification_cm = pd.DataFrame(sum(apt_identification_list), columns=[0, 1], index=[0, 1])
        apt_identification_cm.to_csv(path + "apt_identification.csv", index=False)
        apt_identification_cm_full = sum(apt_identification_full_list)
        apt_identification_cm_full.to_csv(path + "apt_identification_full.csv", index=False)
    print("END")

def compute_result(folder):
    try:
        compute_result_rf(folder)
    except FileNotFoundError as e:
        pass
    try:
        compute_result_oc(folder)
    except FileNotFoundError as e:
        pass



def time_data_rf(result_time_df, path):
    time_df = pd.DataFrame({"train":result_time_df["time_train"], "test":result_time_df["time_apt"]})
    train_mean = mean(time_df["train"])
    train_std = stdev(time_df["train"])
    test_mean = mean(time_df["test"])
    test_std = stdev(time_df["test"])
    time_result = pd.DataFrame([{"Train": '{0:.10f}'.format(train_mean), "Train_std": '{0:.10f}'.format(train_std),
                                 "Test": '{0:.10f}'.format(test_mean), "Test_std": '{0:.10f}'.format(test_std)}])
    if not os.path.exists(path):
        os.makedirs(path)
    time_result.to_csv(path + "time.csv", index=False)



def time_data_oc(result_time_df, path):
    time_list = []
    for index in range(0, 10):
        time_list.append(result_time_df["time_data"].iloc[(index * 6):(index * 6 + 10)].sum())
    time_df = pd.DataFrame(time_list)
    train_mean = mean(time_df["train"])
    train_std = stdev(time_df["train"])
    time_df["test"] = time_df.apply(lambda x: x["test_apt"] + x["test_malware"], 1)
    test_mean = mean(time_df["test"])
    test_std = stdev(time_df["test"])
    lda_mean = mean(result_time_df["lda_time"])
    lda_std = stdev(result_time_df["lda_time"])
    time_result = pd.DataFrame([{"LDA": '{0:.10f}'.format(lda_mean), "LDA_std": '{0:.10f}'.format(lda_std),
                                 "Train": '{0:.10f}'.format(train_mean), "Train_std": '{0:.10f}'.format(train_std),
                                 "Test": '{0:.10f}'.format(test_mean), "Test_std": '{0:.10f}'.format(test_std)}])
    if not os.path.exists(path):
        os.makedirs(path)
    time_result.to_csv(path + "time.csv", index=False)


def time_data(folder):
    try:
        with open(folder + "rf_result.p", "rb") as infile:
            result_rf_full = pickle.load(infile)
        time_data_rf(result_rf_full, folder + "rf/")
    except FileNotFoundError as e:
        pass

    try:
        with open(folder + "oc_result.p", "rb") as infile:
            result_oc_full = pickle.load(infile)
        time_data_oc(result_oc_full, folder + "oc/")
    except FileNotFoundError as e:
        pass
