# coding: utf-8
from os.path import dirname, join
import numpy as np
import pandas as pd
import subprocess
import random

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

from functions import predictivity_classif, simplicity, q_stability, find_bins,\
    extract_rules_from_tree, make_rs_from_r

import warnings
warnings.filterwarnings("ignore")

target_dict = {'crx': 'y',
               'german': 'y',
               'haberman': 'survival',
               'heart': 'y',
               'ionosphere': 'y',
               'bupa': 'selector',
               'wine': 'quality',
               'speaker': 'language',
               'covertype': 'Cover_Type',
               'student': 'atd'}

racine_path = dirname(__file__)
data_path = r'/home/vincent/Documents/Data/Classification/'

pathx = join(racine_path, 'X.csv')
pathx_test = join(racine_path, 'X_test.csv')
pathy = join(racine_path, 'Y.csv')
pathr = join(racine_path, 'main_classif.r')
r_script = '/usr/bin/Rscript'


def load_data(name: str):
    """
    Parameters
    ----------
    name: a chosen data set

    Returns
    -------
    data: a pandas DataFrame
    """
    if name == 'wine':
        data = pd.read_csv(join(data_path, 'Wine/wine.csv'), sep=';')
    elif name == 'speaker':
        data = pd.read_csv(join(data_path, 'Speaker/speaker.csv'))
    elif name == 'covertype':
        data = pd.read_csv(join(data_path, 'CoverType/covertype.csv'))
    elif name == 'student':
        data = pd.read_csv(join(data_path, 'Student/student.csv'))
    else:
        raise ValueError('Not tested dataset')
    return data.dropna()


if __name__ == '__main__':
    test_size = 0.2
    np.random.seed(2020)

    q = 10
    nb_simu = 20
    res_dict = {}
    #  Data parameters
    for data_name in ['speaker', 'student', 'wine', 'covertype']:
        print('')
        print('===== ', data_name.upper(), ' =====')
        res_dict['DT'] = []
        res_dict['RIPPER'] = []
        res_dict['PART'] = []

        dataset = load_data(data_name)
        target = target_dict[data_name]
        y = dataset[target]
        X = dataset.drop(target, axis=1)
        features = X.columns
        X = X[features]
        if data_name == 'student':
            le = preprocessing.LabelEncoder()
            for col in features:
                X[col] = le.fit_transform(X[col])
        if data_name == 'covertype':
            id_kept = random.sample(range(X.shape[0]), 10000)
            X = X.iloc[id_kept]
            y = y.iloc[id_kept]

        kf = KFold(n_splits=nb_simu)
        simu = 0
        for train_index, test_index in kf.split(X):
            # ## Data Generation
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            X_train.to_csv(pathx, index=False)
            y_train.to_csv(pathy, index=False, header=False)
            X_test.to_csv(pathx_test, index=False)
            n_train = len(X_train)
            d = X_train.shape[1]

            sub_id = random.sample(list(range(n_train)), int(n_train / 2))
            sub_id2 = list(filter(lambda i: i not in sub_id, range(n_train)))
            X1 = X_train.iloc[sub_id]
            X2 = X_train.iloc[sub_id2]
            y1 = y_train.iloc[sub_id]
            y2 = y_train.iloc[sub_id2]

            y_train = y_train.values
            y_test = y_test.values
            X_train = X_train.values  # To get only numerical variables
            X_test = X_test.values

            bins_dict = {}
            for k in range(d):
                xcol = X_train[:, k]
                if len(set(xcol)) > q:
                    if type(xcol[0]) == str:
                        bins_dict[features[k]] = sorted(set(xcol))
                    else:
                        var_bins = find_bins(xcol, q)
                        bins_dict[features[k]] = var_bins
                else:
                    bins_dict[features[k]] = sorted(set(xcol))

            with open('output_rfile.txt', 'w') as f:
                subprocess.call([r_script, "--no-save", "--no-restore",
                                 "--verbose", "--vanilla", pathr,
                                 pathx, pathy, pathx_test, 'TRUE'],
                                stdout=f, stderr=subprocess.STDOUT)

            pred_ripper = pd.read_csv(join(racine_path, 'ripper_pred.csv'))['x'].values
            pred_part = pd.read_csv(join(racine_path, 'part_pred.csv'))['x'].values
            rules_ripper = pd.read_csv(join(racine_path, 'ripper_rules.csv'))
            rules_part = pd.read_csv(join(racine_path, 'part_rules.csv'))
            ripper_rs = make_rs_from_r(rules_ripper, features.to_list(), X_train.min(axis=0),
                                       X_train.max(axis=0))
            part_rs = make_rs_from_r(rules_part, features.to_list(), X_train.min(axis=0),
                                     X_train.max(axis=0))

            subsample = min(0.5, (100 + 6 * np.sqrt(len(y_train))) / len(y_train))

            # ## Decision Tree
            tree = DecisionTreeClassifier(max_leaf_nodes=10)
            tree.fit(X_train, y_train)

            tree_rules = extract_rules_from_tree(tree, features, X_train.min(axis=0),
                                                 X_train.max(axis=0), get_leaf=True)

            # ## Errors calculation
            pred_tree = tree.predict(X_test)
            # pred_rulefit = rule_fit.predict(X_test)

            rs_dict = {'Ripper': [], 'Part': [], 'DT': []}

            for sub_x, sub_y in zip([X1, X2], [y1, y2]):
                sub_x.to_csv(pathx, index=False)
                sub_y.to_csv(pathy, index=False)

                with open('output_rfile.txt', 'w') as f:
                    subprocess.call([r_script, "--no-save", "--no-restore",
                                     "--verbose", "--vanilla", pathr,
                                     pathx, pathy, pathx_test, 'FALSE'],
                                    stdout=f, stderr=subprocess.STDOUT)

                rules_ripper = pd.read_csv(join(racine_path, 'ripper_rules.csv'))
                rules_part = pd.read_csv(join(racine_path, 'part_rules.csv'))

                rs_dict['Ripper'] += [make_rs_from_r(rules_ripper, features.to_list(),
                                                     X_train.min(axis=0), X_train.max(axis=0))]
                rs_dict['Part'] += [make_rs_from_r(rules_part, features.to_list(),
                                                   X_train.min(axis=0),
                                                   X_train.max(axis=0))]

                tree = DecisionTreeClassifier(max_leaf_nodes=10)
                tree.fit(X_train, y_train)

                rs_dict['DT'] += [extract_rules_from_tree(tree, features, X_train.min(axis=0),
                                                          X_train.max(axis=0), get_leaf=True)]

            simp = [simplicity(tree_rules), simplicity(ripper_rs), simplicity(part_rs)]
            simp = min(simp) / np.array(simp)
            if simu == 0:
                res_dict['DT'] = [[predictivity_classif(pred_tree, y_test),
                                   q_stability(rs_dict['DT'][0], rs_dict['DT'][1],  X_train,
                                               q=q, bins_dict=bins_dict),
                                   simp[0]]]
                res_dict['RIPPER'] = [[predictivity_classif(pred_ripper, y_test),
                                       q_stability(rs_dict['Ripper'][0], rs_dict['Ripper'][1],
                                                   X_train, q=q, bins_dict=bins_dict),
                                       simp[1]]]
                res_dict['PART'] = [[predictivity_classif(pred_part, y_test),
                                     q_stability(rs_dict['Part'][0], rs_dict['Part'][1],
                                                 X_train, q=q, bins_dict=bins_dict),
                                     simp[2]]]

            else:
                res_dict['DT'] = np.append(res_dict['DT'],
                                           [[predictivity_classif(pred_tree, y_test),
                                             q_stability(rs_dict['DT'][0], rs_dict['DT'][1],
                                                         X_train, q=q, bins_dict=bins_dict),
                                             simp[0]]], axis=0)
                res_dict['RIPPER'] = np.append(res_dict['RIPPER'],
                                               [[predictivity_classif(pred_ripper, y_test),
                                                 q_stability(rs_dict['Ripper'][0],
                                                             rs_dict['Ripper'][1],
                                                             X_train, q=q, bins_dict=bins_dict),
                                                 simp[1]]], axis=0)
                res_dict['PART'] = np.append(res_dict['PART'],
                                             [[predictivity_classif(pred_part, y_test),
                                               q_stability(rs_dict['Part'][0], rs_dict['Part'][1],
                                                           X_train, q=q, bins_dict=bins_dict),
                                               simp[2]]], axis=0)
            simu += 1

        # ## Results.
        print('Predictivity score')
        print('----------------------')
        print('Decision tree predicitivty score:', np.mean(res_dict['DT'][:, 0]))
        print('RIPPER predicitivty score:', np.mean(res_dict['RIPPER'][:, 0]))
        print('PART predicitivty score:', np.mean(res_dict['PART'][:, 0]))
        print('')
        print('q-Stability score')
        print('----------------------')
        print('Decision tree q-Stability score:', np.mean(res_dict['DT'][:, 1]))
        print('RIPPER q-Stability score:', np.mean(res_dict['RIPPER'][:, 1]))
        print('PART q-Stability score:', np.mean(res_dict['PART'][:, 1]))
        print('')
        print('Simplicity score')
        print('----------------------')
        print('Decision tree Simplicity score:', np.mean(res_dict['DT'][:, 2]))
        print('RIPPER Simplicity score:', np.mean(res_dict['RIPPER'][:, 2]))
        print('PART Simplicity score:', np.mean(res_dict['PART'][:, 2]))
