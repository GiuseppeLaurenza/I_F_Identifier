import json
import pickle
import time
import warnings
from collections import defaultdict
from multiprocessing import cpu_count

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from ThresholdRandomForest import ThresholdRandomForest

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
dataset_path = "dataset.csv"
FULL_DATASET = pd.read_csv(dataset_path)
FULL_DATASET["apt"] = FULL_DATASET["apt"].fillna("")
config_oc_path = "selected_param.json"
with open(config_oc_path) as f:
    CONFIG_OC = json.loads(f.read())

cleanDataset = FULL_DATASET[FULL_DATASET["apt"] != ""]
noAptDataset = FULL_DATASET[FULL_DATASET["apt"] == ""]


def test_oc(result_path, dataset):
    apt_list = list(set(dataset['apt']))
    clf = LinearDiscriminantAnalysis(solver='svd')
    lda_time_list = []
    for i in range(0, 10):
        lda_time = time.time()
        X_LDA = pd.DataFrame(clf.fit_transform(dataset.drop(["apt", "md5"], 1), dataset['apt']))
        lda_end_train_time = time.time()
        lda_end_time = lda_end_train_time - lda_time
        lda_time_list.append(lda_end_time)

    X_LDA = X_LDA.add_prefix('col_')
    features_list = X_LDA.columns.values
    df = X_LDA.assign(apt=dataset["apt"]).assign(md5=dataset["md5"])
    lb = LabelBinarizer(neg_label=-1)
    classes = lb.fit_transform(df["apt"])
    binarized_class = pd.DataFrame(classes, columns=lb.classes_)
    df_binarized = pd.concat([df, binarized_class], axis=1, sort=False).assign(apt=dataset["apt"]).reset_index(
        drop=True)
    noAPT_LDA = pd.DataFrame(clf.transform(noAptDataset.drop(["apt", "md5"], 1))).add_prefix("col_").assign(
        apt=-1).assign(
        md5=noAptDataset["md5"])
    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cm_apt_dict = defaultdict(list)
    cm_dual_list = []
    time_train_list = defaultdict(list)
    time_apt_list = defaultdict(list)
    time_dual_list = defaultdict(list)
    for train_index, test_index in kf.split(df_binarized[features_list], dataset['apt']):
        no_apt_test = pd.DataFrame(-1, index=range(noAPT_LDA.shape[0]), columns=apt_list)
        current_global_pred = pd.concat([binarized_class.iloc[test_index], no_apt_test])
        for apt_name in apt_list:
            start_time = time.time()
            X = df_binarized[features_list]
            y = df_binarized[apt_name]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            X_train = X.iloc[train_index][y_train == 1]
            X_test = X.iloc[test_index]
            current_contamination = CONFIG_OC[apt_name][0]
            clf = IsolationForest(contamination=current_contamination, n_estimators=CONFIG_OC[apt_name][1],
                                  random_state=42,
                                  n_jobs=cpu_count() - 1)
            clf.fit(X_train, y_train[y_train == 1])
            end_train_time = time.time()
            end_train = end_train_time - start_time
            time_train_list[apt_name].append(end_train)
            pred = clf.predict(X_test)
            end_test_apt_time = time.time()
            end_test_apt = end_test_apt_time - end_train_time
            time_apt_list[apt_name].append(end_test_apt)
            apt_cm = confusion_matrix(y_test, pred)
            cm_apt_dict[apt_name].append(apt_cm)
            X_test = X.iloc[test_index].append(noAPT_LDA[features_list]).reset_index(drop=True)
            pred = clf.predict(X_test)
            end_test_dual = time.time() - end_test_apt_time
            time_dual_list[apt_name].append(end_test_dual)
            current_global_pred["pred_" + apt_name] = pred
        cm_dual_list.append(current_global_pred)
    result_dict = {"cm_apt_dict": cm_apt_dict, "cm_dual_list": cm_dual_list, "time_train": time_train_list,
                   "time_apt": time_apt_list, "time_dual": time_dual_list, "lda_time": lda_time_list}
    with open(result_path, "wb") as outputfile:
        pickle.dump(result_dict, outputfile)


def test_rf(result_path, dataset):
    X = dataset.drop(["apt"], 1)
    y = dataset['apt']
    lb = LabelBinarizer(neg_label=-1)
    classes = lb.fit_transform(y)
    binarized_class = pd.DataFrame(classes, columns=lb.classes_)
    df_binarized = pd.concat([dataset, binarized_class], axis=1, sort=False)

    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    pred_5_list = []
    pred_10_list = []
    pred_15_list = []
    time_train_list = []
    time_apt_list = []
    for train_index, test_index in kf.split(X, y):
        start_time = time.time()
        y_train, y_test = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].append(
            noAptDataset["apt"]).reset_index(drop=True)
        X_train, X_test = X.iloc[train_index].reset_index(drop=True), X.iloc[test_index].append(
            noAptDataset.drop("apt", axis=1)).reset_index(drop=True)
        clf = ThresholdRandomForest(percentage=0.05, n_estimators=150, random_state=42,
                                    n_jobs=cpu_count() - 1, class_name="apt")
        clf.fit(X_train, y_train)
        end_train_time = time.time()
        end_train = end_train_time - start_time
        time_train_list.append(end_train)
        clf.set_percentage(0.05)
        pred = clf.predict(X_test)
        pred_5_list.append(pred)
        end_test = time.time() - end_train_time
        time_apt_list.append(end_test)
        clf.set_percentage(0.10)
        pred = clf.predict(X_test)
        pred_10_list.append(pred)
        clf.set_percentage(0.15)
        pred = clf.predict(X_test)
        pred_15_list.append(pred)
    result_dict = {"pred_5": pred_5_list, "pred_10": pred_10_list, "pred_15": pred_15_list,
                   "time_apt": time_apt_list, "time_train": time_train_list, "df_binarized": df_binarized}
    with open(result_path, "wb") as outputfile:
        pickle.dump(result_dict, outputfile)


def main():
    kf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    X = cleanDataset.drop(["apt", "md5"], 1)
    y = cleanDataset['apt']
    y_true = []
    y_pred = []
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

    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred, labels=clf.classes_)
    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index, columns=clf.classes_).transpose()

    print("Started Analysis with 0.95")
    class_95 = list(class_report_df[(class_report_df["precision"] > 0.95) & (class_report_df["recall"] > 0.95)].index)
    print(class_95)
    print("Start Isolation Forest")
    test_oc(result_path="oneclass_result_" + str(len(class_95)) + ".p",
            dataset=FULL_DATASET[FULL_DATASET["apt"].isin(class_95)].reset_index(drop=True))
    print("Start Random Forest")
    test_rf(result_path="thresholdrf_result_" + str(len(class_95)) + ".p",
            dataset=FULL_DATASET[FULL_DATASET["apt"].isin(class_95)].reset_index(drop=True))

    print("Started Analysis with 0.90")
    class_90 = list(class_report_df[(class_report_df["precision"] > 0.90) & (class_report_df["recall"] > 0.90)].index)
    print(class_90)
    # class_90.append("")
    print("Start Isolation Forest")
    test_oc(result_path="oneclass_result_" + str(len(class_90)) + ".p",
            dataset=FULL_DATASET[FULL_DATASET["apt"].isin(class_90)].reset_index(drop=True))
    print("Start Random Forest")
    test_rf(result_path="thresholdrf_result_" + str(len(class_90)) + ".p",
            dataset=FULL_DATASET[FULL_DATASET["apt"].isin(class_90)].reset_index(drop=True))

    print("Started Analysis with Full Dataset")
    class_full = list(set(cleanDataset["apt"]))
    print(class_full)
    print("Start Isolation Forest")
    test_oc(result_path="oneclass_result_all.p",
            dataset=cleanDataset.reset_index(drop=True))
    print("Start Random Forest")
    test_rf(result_path="thresholdrf_result_all.p",
            dataset=cleanDataset.reset_index(drop=True))

    print("###############END############")
    # test_oc()
    # test_rf()


main()
