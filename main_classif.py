# coding: utf-8
from os.path import dirname, join
import numpy as np
import pandas as pd
import subprocess
import random

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import rulefit
# 'Install the package rulefit from Christophe Molnar GitHub with the command'
# 'pip install git+git://github.com/christophM/rulefit.git')

from functions import predictivity_classif, simplicity, q_stability, find_bins,\
    extract_rules_from_tree, extract_rules_rulefit, make_rs_from_r


target_dict = {'crx': 'y',
               'german': 'y',
               'haberman': 'survival',
               'heart': 'y',
               'ionosphere': 'y',
               'bupa': 'selector'}

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
    if name == 'crx':
        data = pd.read_csv(join(data_path, 'Credit Approval/crx.csv'))
        data['Att1'] = [1 if x == 'b' else 0 for x in data['Att1'].values]
        data['Att4'] = [1 if x == 'y' else 0 for x in data['Att4'].values]
        data['Att5'] = [1 if x == 'g' else 0 for x in data['Att5'].values]
        data['Att9'] = [1 if x == 't' else 0 for x in data['Att9'].values]
        data['Att10'] = [1 if x == 't' else 0 for x in data['Att10'].values]
        data['Att12'] = [1 if x == 't' else 0 for x in data['Att12'].values]
        data['Att13'] = [1 if x == 'g' else 0 for x in data['Att13'].values]
        # data['y'] = [1 if x == '+' else 0 for x in data['y'].values]

        data = data.drop(['Att6', 'Att7'], axis=1)
    elif name == 'german':
        data = pd.read_csv(join(data_path, 'Credit German/german_num.csv'))
    elif name == 'haberman':
        data = pd.read_csv(join(data_path, 'Haberman/haberman.csv'))
    elif name == 'heart':
        data = pd.read_csv(join(data_path, 'Heart Statlog/heart.csv'))
    elif name == 'ionosphere':
        data = pd.read_csv(join(data_path, 'Ionosphere/ionosphere.csv'))
    elif name == 'bupa':
        data = pd.read_csv(join(data_path, 'Liver Disorders/bupa.csv'))
    else:
        raise ValueError('Not tested dataset')
    return data.dropna()


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    test_size = 0.3

    # RF parameters
    tree_size = 4  # number of leaves by tree
    max_rules = 10000  # total number of rules generated from tree ensembles
    nb_estimator = int(np.ceil(max_rules / tree_size))  # Number of tree

    # AdBoost and GradientBoosting
    learning_rate = 0.2

    # Covering parameters
    alpha = 1. / 2 - 1 / 100.
    gamma = 0.95
    lmax = 2

    q = 10

    #  Data parameters
    for data_name in ['crx', 'german', 'haberman', 'heart', 'inosphere', 'bupa']:
        print('')
        print('===== ', data_name.upper(), ' =====')

        # ## Data Generation
        dataset = load_data(data_name)
        target = target_dict[data_name]
        y = dataset[target]
        X = dataset.drop(target, axis=1)
        features = X.columns
        X = X[features]

        # ### Splitting data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=seed)
        if test_size == 0.0:
            X_test = X_train
            y_test = y_train

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

        # pred_sirus = pd.read_csv(join(racine_path, 'sirus_pred.csv'))['x'].values
        # pred_nh = pd.read_csv(join(racine_path, 'nh_pred.csv'))['x'].values
        # rules_sirus = pd.read_csv(join(racine_path, 'sirus_rules.csv'))
        # rules_nh = pd.read_csv(join(racine_path, 'nh_rules.csv'))
        #
        # sirus_rs = make_rs_from_r(rules_sirus, features.to_list(), X_train.min(axis=0),
        #                           X_train.max(axis=0))
        # nh_rs = make_rs_from_r(rules_nh, features.to_list(), X_train.min(axis=0),
        #                        X_train.max(axis=0))

        subsample = min(0.5, (100 + 6 * np.sqrt(len(y_train))) / len(y_train))

        # ## Decision Tree
        tree = DecisionTreeClassifier(max_leaf_nodes=20,  # tree_size,
                                      random_state=seed)
        tree.fit(X_train, y_train)

        tree_rules = extract_rules_from_tree(tree, features, X_train.min(axis=0),
                                             X_train.max(axis=0))

        # # ## RuleFit
        # rule_fit = rulefit.RuleFit(tree_size=tree_size,
        #                            max_rules=max_rules,
        #                            random_state=seed,
        #                            max_iter=2000)
        # rule_fit.fit(X_train, y_train)
        #
        # # ### RuleFit rules part
        # rules = rule_fit.get_rules()
        # rules = rules[rules.coef != 0].sort_values(by="support")
        # rules = rules.loc[rules['type'] == 'rule']
        #
        # # ### RuleFit linear part
        # lin = rule_fit.get_rules()
        # lin = lin[lin.coef != 0].sort_values(by="support")
        # lin = lin.loc[lin['type'] == 'linear']
        #
        # rulefit_rules = extract_rules_rulefit(rules, features, X_train.min(axis=0),
        #                                       X_train.max(axis=0))

        # ## Errors calculation
        pred_tree = tree.predict(X_test)
        # pred_rulefit = rule_fit.predict(X_test)

        rs_dict = {  # 'Sirus': [], 'NH': [],
                   'Ripper': [], 'Part': [], 'DT': [],
            # 'RuleFit': []
        }
        for sub_x, sub_y in zip([X1, X2], [y1, y2]):
            sub_x.to_csv(pathx, index=False)
            sub_y.to_csv(pathy, index=False)

            with open('output_rfile.txt', 'w') as f:
                subprocess.call([r_script, "--no-save", "--no-restore",
                                 "--verbose", "--vanilla", pathr,
                                 pathx, pathy, pathx_test, 'FALSE'],
                                stdout=f, stderr=subprocess.STDOUT)

            # rules_sirus = pd.read_csv(join(racine_path, 'sirus_rules.csv'))
            # rules_nh = pd.read_csv(join(racine_path, 'nh_rules.csv'))
            rules_ripper = pd.read_csv(join(racine_path, 'ripper_rules.csv'))
            rules_part = pd.read_csv(join(racine_path, 'part_rules.csv'))

            # rs_dict['Sirus'] += [make_rs_from_r(rules_sirus, features.to_list(),
            #                                     X_train.min(axis=0), X_train.max(axis=0))]
            # rs_dict['NH'] += [make_rs_from_r(rules_nh, features.to_list(), X_train.min(axis=0),
            #                                  X_train.max(axis=0))]
            rs_dict['Ripper'] += [make_rs_from_r(rules_ripper, features.to_list(),
                                                 X_train.min(axis=0), X_train.max(axis=0))]
            rs_dict['Part'] += [make_rs_from_r(rules_part, features.to_list(), X_train.min(axis=0),
                                               X_train.max(axis=0))]

            tree = DecisionTreeClassifier(max_leaf_nodes=20,  # tree_size,
                                          random_state=seed)
            tree.fit(X_train, y_train)

            rs_dict['DT'] += [extract_rules_from_tree(tree, features, X_train.min(axis=0),
                                                      X_train.max(axis=0))]

            # rule_fit = rulefit.RuleFit(tree_size=tree_size,
            #                            max_rules=max_rules,
            #                            random_state=seed,
            #                            max_iter=2000)
            # rule_fit.fit(X_train, y_train)
            #
            # # ### RuleFit rules part
            # rules = rule_fit.get_rules()
            # rules = rules[rules.coef != 0].sort_values(by="support")
            # rules = rules.loc[rules['type'] == 'rule']
            #
            # rs_dict['RuleFit'] += [extract_rules_rulefit(rules, features, X_train.min(axis=0),
            #                                              X_train.max(axis=0))]

        # ## Results.
        print('Predictivity score')
        print('----------------------')
        print('Decision tree predicitivty score:', predictivity_classif(pred_tree, y_test))
        # print('RuleFit predicitivty score:', predictivity(pred_rulefit, y_test))
        # print('SIRUS predicitivty score:', predictivity(pred_sirus, y_test))
        # print('NH predicitivty score:', predictivity(pred_nh, y_test))
        print('RIPPER predicitivty score:', predictivity_classif(pred_ripper, y_test))
        print('PART predicitivty score:', predictivity_classif(pred_part, y_test))
        print('')
        print('q-Stability score')
        print('----------------------')
        print('Decision tree stability score:', q_stability(rs_dict['DT'][0], rs_dict['DT'][1],
                                                            X_train, q=q, bins_dict=bins_dict))
        # print('RuleFit stability score:', q_stability(rs_dict['RuleFit'], q=q,
        #                                               bins_dict=bins_dict))
        # print('SIRUS stability score:', q_stability(rs_dict['Sirus'], q=q,
        #                                             bins_dict=bins_dict))
        # print('NH stability score:', q_stability(rs_dict['NH'], q=q,
        #                                          bins_dict=bins_dict))
        print('RIPPER stability score:', q_stability(rs_dict['Ripper'][0], rs_dict['Ripper'][1],
                                                     X_train, q=q, bins_dict=bins_dict))
        print('PART stability score:', q_stability(rs_dict['Part'][0], rs_dict['Part'][1],
                                                   X_train, q=q, bins_dict=bins_dict))
        print('')
        print('Simplicity score')
        print('----------------------')
        print('Decision tree simplicity score:', simplicity(tree_rules))
        # print('RuleFit simplicity score:', simplicity(rulefit_rules))
        # print('Linear relation:', len(lin))
        # print('SIRUS simplicity score:', simplicity(sirus_rs))
        # print('NH simplicity score:', simplicity(nh_rs))
        print('RIPPER stability score:', simplicity(ripper_rs))
        print('PART stability score:', simplicity(part_rs))
