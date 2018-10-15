import logging
from collections import Counter
from time import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class ThresholdRandomForest:

    def __sample_confidence_test(self, leaves, sample_leaves):
        '''Compute the confidence of a single sample from
        the leaves dataframe'''
        self.logger.debug("Computation confidence: " + str(sample_leaves))
        confidence_list = []
        for elem in range(0, self.n_estimators):
            confidence_list.append(leaves[elem][int(sample_leaves[elem])])
        result = pd.DataFrame(confidence_list).fillna(0).mean(0)
        result = result.append(pd.Series({"md5": sample_leaves["md5"]}))
        self.logger.debug("End computation")
        return result

    def __compute_confidence_dataframe(self, test):
        '''Compute the dataframe of confidences'''
        self.logger.info("Start confidence dataframe generation")
        start_time = time()
        if isinstance(test, pd.Series):
            test = pd.DataFrame(test).transpose()
        test_leaves = pd.DataFrame(self.rf_model.apply(
            test.drop(["md5", "index"], 1, errors="ignore")))
        test_leaves = test_leaves.assign(md5=test["md5"])
        result = []
        for elem_tuple in test_leaves.iterrows():
            elem = elem_tuple[1]
            temp = self.__sample_confidence_test(self.tree_models, elem)
            result.append(temp)
        result_df = pd.DataFrame(result).fillna(0)
        elapsed_time = time() - start_time
        self.logger.info("Confidence Dataframe generated: " + str(elapsed_time))
        return result_df

    def __sample_confidence_train(self, leaves, sample_leaves, current_class):
        '''Compute the confidence of a single sample from
        the leaves dataframe for training'''
        self.logger.debug("Computation confidence: " + str(sample_leaves))
        confidence = []
        for elem in range(0, self.n_estimators):
            new_list = leaves.loc[leaves[elem] ==
                                  sample_leaves[elem]]["label"].tolist()
            counter = Counter(new_list)
            current_confidence = (counter.get(
                current_class, 0) / (1.0 * len(new_list)))
            confidence.append(current_confidence)
        self.logger.debug("End computation")
        return confidence

    def __leave_one_out(self, index, X, y):
        '''Generate confidence through leave one out methodology'''
        self.logger.debug("Leave one out with element " + str(index))
        start_time = time()
        current_row = X.ix[index]
        current_class = y.ix[index]
        X_train = X.drop(index)
        y_train = y.drop(index)
        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators, criterion="gini", n_jobs=self.n_jobs, random_state=self.random_state)
        model = classifier.fit(X_train.drop(["md5"], axis=1, errors="ignore"), y_train)
        leaves = pd.DataFrame(model.apply(X_train.drop(["md5"], axis=1, errors="ignore")))
        leaves = leaves.assign(label=y)
        sample_leaves = pd.DataFrame(model.apply(current_row.drop(
            ["md5"], errors="ignore").values.reshape(1, -1))).values.tolist()[0]
        confidence = self.__sample_confidence_train(leaves, sample_leaves, current_class)
        elapsed_time = time() - start_time
        self.logger.debug("Leave one out: " + str(elapsed_time))
        return confidence

    def __get_class_confidence(self, X, y):
        '''Compute the average class confidence through
        leave one out methodology'''
        self.logger.info("Starting computation of confidences of classes")
        start_time = time()
        partial_time = time()
        self.logger.info("Starting leave one out phase")
        confidence_list = []
        for i in range(0, X.shape[0]):
            elem = self.__leave_one_out(i, X, y)
            confidence_list.append(elem)
        elapsed_time = time() - partial_time
        self.logger.info("End loo phase " + str(elapsed_time))
        averages = [sum(suby) / len(suby) for suby in confidence_list]
        averages_df = pd.DataFrame(averages)
        averages_df = averages_df.assign(classes=y.values)
        class_confidences = dict()
        for elem in averages_df.groupby("classes"):
            if elem[1].empty:
                class_confidences[elem[0]] = 0
            else:
                class_confidences[elem[0]] = sum(elem[1][0]) / (1.0 * len(elem[1][0]))
        elapsed_time = time() - start_time
        self.logger.info("Ended computation: " + str(elapsed_time))
        self.class_confidences = class_confidences

    def __generate_confidence_model(self, x, y):
        '''Generate confidence model'''
        classifier = RandomForestClassifier(
            n_estimators=self.n_estimators, criterion="gini", n_jobs=self.n_jobs, random_state=self.random_state)
        model = classifier.fit(x.drop(["md5", "index"], 1, errors="ignore"), y)
        leaves = pd.DataFrame(model.apply(x.drop(["md5", "index"], 1, errors="ignore")))
        assign_args = {self.class_name: y}
        leaves = leaves.assign(**assign_args)
        leaves_dict = dict()
        for index in range(0, leaves.shape[1] - 1):
            group = leaves[[index, self.class_name]].groupby(index)
            node_dict = dict()
            for node in group:
                counter = Counter(node[1][self.class_name])
                node_dict[node[0]] = dict(
                    [(elem, counter[elem] / float(len(node[1][self.class_name])) * 1.0) for elem in counter])
            leaves_dict[index] = node_dict
        result = dict()
        self.rf_model = model
        self.tree_models = leaves_dict
        return result

    def fit(self, x, y):
        self.__get_class_confidence(x, y)
        self.__generate_confidence_model(x, y)

    def apply(self, x):
        return self.__compute_confidence_dataframe(x)

    def get_percentage(self):
        return self.percentage

    def set_percentage(self, percentage):
        self.percentage = percentage

    def predict(self, x):
        result_prediction = self.__compute_confidence_dataframe(x)
        cols = list(result_prediction.columns.values)
        try:
            cols.remove("md5")
        except Exception as e:
            pass
        percentage_function = lambda x: (x - (x * self.percentage))
        percentage_table = pd.DataFrame.from_records(self.class_confidences, index=[0]).apply(percentage_function)
        result = pd.DataFrame()
        for apt_name in cols:
            result[apt_name] = result_prediction[apt_name] - percentage_table[apt_name][0]
        result = pd.DataFrame(data=np.where(result > 0, 1, -1), columns=cols)
        result["md5"] = x["md5"]
        return result

    def __init__(self, n_estimators=100, percentage=0.05, random_state=1, n_jobs=1, debug=0,
                 class_name="current_class"):
        self.n_estimators = n_estimators
        self.percentage = percentage
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        self.class_name = class_name
        logging.basicConfig(format='%(asctime)s %(message)s')
        if debug == "1":
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
