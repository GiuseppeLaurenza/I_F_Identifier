import json
import logging
import os
import pickle
import time
from multiprocessing import cpu_count
from statistics import mean, variance

import numpy as np
import pandas as pd
from pycm import ConfusionMatrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from ThresholdRandomForest import ThresholdRandomForest


def compute_result_rf(result_file, destination):
    with open(result_file, "rb") as infile:
        data = pickle.load(infile)
    for dirName in ["5/","10/","15/"]:
        if not os.path.exists(destination+dirName):
            os.mkdir(destination+dirName)
        result = compute_result(data["pred_"+dirName[:-1]])
        result["detection_cm"].to_csv(destination+dirName+"apt_detection.csv",index=False)
        result["apt_identification_cm"].to_csv(destination+dirName+"apt_identification.csv", index=False)
        result["apt_identification_cm_full"].to_csv(destination+dirName+"apt_identification_full.csv", index=False)


def compute_result_if(result_file, destination):
    with open(result_file, "rb") as infile:
        data = pickle.load(infile)
    result = compute_result(data["pred"])
    result["detection_cm"].to_csv(destination+"apt_detection.csv",index=False)
    result["apt_identification_cm"].to_csv(destination+"apt_identification.csv", index=False)
    result["apt_identification_cm_full"].to_csv(destination+"apt_identification_full.csv", index=False)

def compute_result(result_data):
    apt_detection_list = []
    apt_identification_full_list = []
    apt_identification_list = []
    for elem in result_data:
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
    apt_identification_cm = pd.DataFrame(sum(apt_identification_list), columns=[0, 1], index=[0, 1])
    apt_identification_cm_full = sum(apt_identification_full_list)
    return({"detection_cm":detection_cm, "apt_identification_cm":apt_identification_cm, "apt_identification_cm_full":apt_identification_cm_full})

def compute_time_rf(result_file, destination):
    with open(result_file,"rb") as infile:
        time_data = pickle.load(infile)["time_data"]
    train_mean = mean(time_data["train"])
    train_variance = variance(time_data["train"])
    test_mean = mean(time_data["test"])
    test_variance = variance(time_data["test"])
    time_result = pd.DataFrame([{"Train": '{0:.10f}'.format(train_mean), "Train_var": '{0:.10f}'.format(train_variance),
                   "Test": '{0:.10f}'.format(test_mean), "Test_var": '{0:.10f}'.format(test_variance)}])
    time_result.to_csv(destination+"time.csv",index=False)

def compute_time_if(result_file, destination):
    with open(result_file,"rb") as infile:
        time_data = pickle.load(infile)
    time_list = []
    for index in range(0,10):
        time_list.append(time_data["time_data"].iloc[(index*6):(index*6+10)].sum())
    time_df = pd.DataFrame(time_list)
    train_mean = mean(time_df["train"])
    train_variance = variance(time_df["train"])
    time_df["test"] = time_df.apply(lambda x: x["test_apt"]+x["test_malware"],1)
    test_mean = mean(time_df["test"])
    test_variance = variance(time_df["test"])
    lda_mean = mean(time_data["lda_time"])
    lda_variance = variance(time_data["lda_time"])

    time_result = pd.DataFrame([{"LDA": '{0:.10f}'.format(lda_mean), "LDA_var": '{0:.10f}'.format(lda_variance),
                   "Train": '{0:.10f}'.format(train_mean), "Train_var": '{0:.10f}'.format(train_variance),
                   "Test": '{0:.10f}'.format(test_mean), "Test_var": '{0:.10f}'.format(test_variance)}])
    time_result.to_csv(destination+"time.csv",index=False)



def best_class(dataset):
    X = dataset.drop(["md5", "apt"], 1)
    y = dataset['apt']
    y_true = []
    y_pred = []
    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    logging.info("Checking best six classes")
    imp_feat = []
    for train_index, test_index in kf.split(X, y):
        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        clf = RandomForestClassifier(n_estimators=150, random_state=1, n_jobs=cpu_count() - 1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        y_true.extend(y_test)
        y_pred.extend(pred)
        feature_importances = pd.DataFrame(clf.feature_importances_,
                                           index=X_train.columns,
                                           columns=['importance']).sort_values('importance', ascending=False)
        imp_feat.append(feature_importances[feature_importances["importance"] > 0])
    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred, labels=clf.classes_)
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index, columns=clf.classes_).transpose().sort_values(["precision", "recall"],
                                                                               ascending=[0, 0])
    selected_classes = list(
        class_report_df[(class_report_df["precision"] > 0.95) & (class_report_df["recall"] > 0.95)].index)
    return selected_classes


def test_if(apt_dataset, malware_dataset, parameters, destination):
    logging.info("One class test - output in " + str(destination))
    apt_list = list(set(apt_dataset['apt']))
    clf = LinearDiscriminantAnalysis(solver='svd', )
    lda_time_list = []
    logging.info("LDA Phase")
    for i in range(0, 10):
        lda_time = time.time()
        X_LDA = pd.DataFrame(clf.fit_transform(apt_dataset.drop(["apt", "md5"], 1), apt_dataset['apt']))
        lda_end_train_time = time.time()
        lda_end_time = lda_end_train_time - lda_time
        lda_time_list.append(lda_end_time)
    X_LDA = pd.DataFrame(clf.fit_transform(apt_dataset.drop(["apt", "md5"], 1), apt_dataset['apt']))
    X_LDA = X_LDA.add_prefix('col_')
    features_list = X_LDA.columns.values
    df = X_LDA.assign(apt=apt_dataset["apt"])
    logging.info("Binarizing Label Phase")
    lb = LabelBinarizer(neg_label=-1)
    classes = lb.fit_transform(df["apt"])
    binarized_class = pd.DataFrame(classes, columns=lb.classes_)
    apt_binarized = pd.concat([df, binarized_class], axis=1, sort=False).assign(apt=apt_dataset["apt"]).reset_index(
        drop=True)

    noAPT_LDA = pd.DataFrame(clf.transform(malware_dataset.drop(["md5"], 1))).add_prefix("col_")
    # noAPT_LDA = pd.concat([noAPT_LDA, pd.DataFrame(columns=lb.classes_)], sort=False).fillna(-1)
    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    model_list = []
    time_data = pd.DataFrame(columns=["apt_name","train","test_apt", "test_malware"])
    prediction_list = []
    result_list = []
    logging.info("Test Beginning")
    for train_index, test_index in kf.split(apt_binarized, apt_dataset["apt"]):
        X = apt_binarized[features_list]
        y = apt_binarized[apt_list]
        pred_df = pd.DataFrame(columns=apt_list)
        res_df = pd.DataFrame(columns=apt_list)
        current_model_dict = dict()
        current_pred_dict = dict()
        for apt_name in apt_list:
            start_time = time.time()
            y_train = y[apt_name].iloc[train_index]
            y_test = y[apt_name].iloc[test_index]
            apt_pred_dict = dict()
            apt_pred_dict["apt"] = y_test
            X_train = X.iloc[train_index][y_train == 1]
            X_test = X.iloc[test_index]
            current_contamination = parameters[apt_name][0]
            clf = IsolationForest(contamination=current_contamination, n_estimators=parameters[apt_name][1],
                                  random_state=42, behaviour="new",
                                  n_jobs=cpu_count() - 1)
            clf.fit(X_train, y_train[y_train == 1])
            end_train_time = time.time()
            end_train = end_train_time - start_time
            current_model_dict[apt_name] = clf
            pred_apt = clf.predict(X_test)

            apt_pred_dict["pred_apt"] = pred_apt
            end_test_apt_time = time.time()
            end_test_apt = end_test_apt_time - end_train_time
            pred_malware = clf.predict(noAPT_LDA)
            apt_pred_dict["pred_malware"] = pred_malware
            pred_df[apt_name] = np.append(pred_apt, pred_malware)
            res_df[apt_name] = np.append(y_test,[-1] * len(pred_malware))
            current_pred_dict[apt_name] = apt_pred_dict
            end_test_malware_time = time.time()
            end_test_malware = end_test_malware_time - end_test_apt_time
            time_data = time_data.append({"apt_name":apt_name,"train":end_train,"test_apt":end_test_apt, "test_malware":end_test_malware}, ignore_index=True)

        model_list.append(current_model_dict)
        prediction_list.append(current_pred_dict)
        result_list.append({"pred": pred_df, "res": res_df})

    output_dict = {"models":model_list, "pred":result_list, "lda_time":lda_time_list, "time_data":time_data}

    logging.info("Store result")
    with open(destination+"if_result.p","wb") as outfile:
        pickle.dump(output_dict, outfile)
    compute_result_if(destination+"if_result.p", destination)
    compute_time_if(destination+"if_result.p", destination)

    logging.info("One class test completed")


def test_rf(apt_dataset, malware_dataset, destination):
    X = apt_dataset.drop(["apt"], 1)
    y = apt_dataset['apt']

    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    pred_5_list = []
    pred_10_list = []
    pred_15_list = []
    model_list = []
    time_data = pd.DataFrame(columns=["train", "test"])
    for train_index, test_index in kf.split(X, y):
        start_time = time.time()
        y_train, y_test = y.iloc[train_index].reset_index(drop=True), np.append(y.iloc[test_index],[''] * malware_dataset.shape[0])
        X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].append(
            malware_dataset).reset_index(drop=True)
        lb = LabelBinarizer(neg_label=-1)
        classes = lb.fit_transform(y_test)
        res_df = pd.DataFrame(classes, columns=lb.classes_)
        res_df = res_df.drop("",1)
        clf = ThresholdRandomForest(percentage=0.05, n_estimators=150, random_state=1,
                                    n_jobs=cpu_count() - 1, class_name="apt")
        clf.fit(X_train, y_train)
        clf.apply(X_test)
        end_train_time = time.time()
        end_train = end_train_time - start_time
        clf.set_percentage(0.05)
        # clf.empty_confidence_dataframe()
        pred = clf.predict(X_test)
        pred_5_list.append({"pred": pred.drop("md5",1), "res": res_df})
        end_test_time = time.time()
        end_test = end_test_time - end_train_time
        time_data = time_data.append({"train":end_train,"test":end_test}, ignore_index=True)
        clf.set_percentage(0.10)
        pred = clf.predict(X_test)
        pred_10_list.append({"pred": pred.drop("md5",1), "res": res_df})
        clf.set_percentage(0.15)
        pred = clf.predict(X_test)
        pred_15_list.append({"pred": pred.drop("md5",1), "res": res_df})
        clf.empty_confidence_dataframe()
        model_list.append(clf)

    output_dict = {"model_list":model_list,"pred_5": pred_5_list, "pred_10": pred_10_list, "pred_15": pred_15_list, "time_data":time_data}
    logging.info("Store result")
    with open(destination+"rf_result.p","wb") as outfile:
        pickle.dump(output_dict, outfile)
    compute_result_rf(destination+"rf_result.p", destination)
    compute_time_rf(destination+"rf_result.p", destination)

def main():
    with open("selected_columns.json", "r") as infile:
        selected_columns = json.load(infile)

    with open("selected_class.json", "r") as infile:
        selected_class = json.load(infile)

    # selected_class = ['Volatile Cedar', 'Shiqiang', 'Violin Panda']


    with open("selected_param.json","r") as infile:
        parameters = json.load(infile)

    apt_dataset = pd.read_hdf("malware_apt.h5")[selected_columns]
    selected_columns.remove("apt")
    malware_dataset = pd.read_hdf("malware_non_apt.h5")[selected_columns]

    test_if(apt_dataset[apt_dataset["apt"].isin(selected_class)].reset_index(drop=True), malware_dataset, parameters, "if_result_best_classes/")
    test_rf(apt_dataset[apt_dataset["apt"].isin(selected_class)].reset_index(drop=True), malware_dataset, "trf_result/")

    test_if(apt_dataset, malware_dataset, parameters, "if_result/")
    test_rf(apt_dataset, malware_dataset, "trf_result/")

main()
