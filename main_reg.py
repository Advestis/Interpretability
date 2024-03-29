# coding: utf-8
from os.path import dirname, join
import numpy as np
import pandas as pd
import subprocess
import random

# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

import rulefit
# 'Install the package rulefit from Christophe Molnar GitHub with the command'
# 'pip install git+git://github.com/christophM/rulefit.git')

import CoveringAlgorithm.CA as CA
from CoveringAlgorithm.functions import make_rs_from_r, extract_rules_rulefit
from Data.load_data import load_data, target_dict

# 'Install the package CoveringAlgorithm from Vincent Margot GitHub with the command'
# 'pip install git+git://github.com/VMargot/CoveringAlgorithm.git')

# import RIPE
# 'Install the package CoveringAlgorithm from Vincent Margot GitHub with the command'
# 'pip install git+git://github.com/VincentM/RIPE.git')

from functions import predictivity, simplicity, q_stability, find_bins

from rule.rule_utils import extract_rules_from_tree

import warnings
warnings.filterwarnings("ignore")

racine_path = dirname(__file__)

pathx = join(racine_path, 'X.csv')
pathx_test = join(racine_path, 'X_test.csv')
pathy = join(racine_path, 'Y.csv')
pathr = join(racine_path, 'main_reg.r')
r_script = '/usr/bin/Rscript'


if __name__ == '__main__':
    test_size = 0.2
    np.random.seed(2021)

    # RF parameters
    tree_size = 4  # number of leaves by tree
    max_rules = 2000  # total number of rules generated from tree ensembles
    nb_estimator = int(np.ceil(max_rules / tree_size))  # Number of tree

    # GradientBoosting
    learning_rate = 0.1

    # Covering parameters
    alpha = 1. / 2 - 1 / 100.
    gamma = 0.95
    lmax = 2

    q = 10
    nb_simu = 10
    res_dict = {}
    #  Data parameters
    for data_name in ['prostate', 'diabetes', 'ozone', 'machine', 'mpg',
                      'boston', 'student_por', 'abalone']:
        print('')
        print('===== ', data_name.upper(), ' =====')

        res_dict['DT'] = []
        res_dict['CA_RF'] = []
        res_dict['CA_GB'] = []
        res_dict['RuleFit'] = []
        res_dict['Sirus'] = []
        res_dict['NH'] = []

        dataset = load_data(data_name, racine_path + '/data/regression/')
        target = target_dict[data_name]
        y = dataset[target].astype('float')
        X = dataset.drop(target, axis=1)
        features = X.describe().columns
        X = X[features]

        kf = KFold(n_splits=nb_simu)
        simu = 0
        for train_index, test_index in kf.split(X):
            # ## Data Generation
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            deno_mse = np.mean((y_test - np.mean(y_train)) ** 2)

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
                                 pathx, pathy, pathx_test],
                                stdout=f, stderr=subprocess.STDOUT)

            pred_sirus = pd.read_csv(join(racine_path, 'sirus_pred.csv'))['x'].values
            pred_nh = pd.read_csv(join(racine_path, 'nh_pred.csv'))['x'].values
            rules_sirus = pd.read_csv(join(racine_path, 'sirus_rules.csv'))
            rules_nh = pd.read_csv(join(racine_path, 'nh_rules.csv'))

            sirus_rs = make_rs_from_r(rules_sirus, features.to_list(), X_train.min(axis=0),
                                      X_train.max(axis=0))
            nh_rs = make_rs_from_r(rules_nh, features.to_list(), X_train.min(axis=0),
                                   X_train.max(axis=0))

            subsample = min(0.5, (100 + 6 * np.sqrt(len(y_train))) / len(y_train))

            # ## Decision Tree
            tree = DecisionTreeRegressor(max_leaf_nodes=10)
            tree.fit(X_train, y_train)

            tree_rules = extract_rules_from_tree(tree, xmins=X_train.min(axis=0), xmaxs=X_train.max(axis=0),
                                                 features_names=features, get_leaf=True)

            # ## Covering Algorithm RandomForest
            ca_rf = CA.CA(alpha=alpha, gamma=gamma,
                          tree_size=tree_size,
                          max_rules=max_rules,
                          generator_func=RandomForestRegressor,
                          lmax=lmax)
            ca_rf.fit(xs=X_train, y=y_train, features=features)

            # ## Covering Algorithm GradientBoosting
            ca_gb = CA.CA(alpha=alpha, gamma=gamma,
                          tree_size=tree_size,
                          max_rules=max_rules,
                          generator_func=GradientBoostingRegressor,
                          lmax=lmax)
            ca_gb.fit(xs=X_train, y=y_train, features=features)

            # ## RuleFit
            rule_fit = rulefit.RuleFit(tree_size=tree_size,
                                       max_rules=max_rules)
            rule_fit.fit(X_train, y_train)

            # ### RuleFit rules part
            rules = rule_fit.get_rules()
            rules = rules[rules.coef != 0].sort_values(by="support")
            rules = rules.loc[rules['type'] == 'rule']
            # ### RuleFit linear part
            lin = rule_fit.get_rules()
            lin = lin[lin.coef != 0].sort_values(by="support")
            lin = lin.loc[lin['type'] == 'linear']

            rulefit_rules = extract_rules_rulefit(rules, features, X_train.min(axis=0),
                                                  X_train.max(axis=0))

            # # ## RIPE
            # ripe = RIPE.Learning(nb_bucket=q, cp=lmax, intermax=gamma)
            # ripe.fit(X, y)

            # ## Errors calculation
            pred_tree = tree.predict(X_test)
            pred_CA_rf = ca_rf.predict(X_test)
            pred_CA_gb = ca_gb.predict(X_test)
            # pred_CA_ad = ca_ad.predict(X_test)
            pred_rulefit = rule_fit.predict(X_test)
            # pred_ripe = ripe.predict(X_test)

            simp = [simplicity(tree_rules),
                    simplicity(rulefit_rules),
                    simplicity(sirus_rs),
                    simplicity(nh_rs),
                    simplicity(ca_rf.selected_rs),
                    simplicity(ca_gb.selected_rs)
                    ]
            # sum(ripe.selected_rs.get_rules_param('cp'))]
            simp = min(simp) / np.array(simp)

            rs_dict = {'Sirus': [],
                       'NH': [],
                       'DT': [],
                       'RuleFit': [],
                       'CA_GB': [],
                       'CA_RF': [],
                       'RIPE': []}
            for sub_x, sub_y in zip([X1, X2], [y1, y2]):
                sub_x.to_csv(pathx, index=False)
                sub_y.to_csv(pathy, index=False)

                with open('output_rfile.txt', 'w') as f:
                    subprocess.call([r_script, "--no-save", "--no-restore",
                                     "--verbose", "--vanilla", pathr,
                                     pathx, pathy, pathx_test, 'FALSE'],
                                    stdout=f, stderr=subprocess.STDOUT)

                rules_sirus = pd.read_csv(join(racine_path, 'sirus_rules.csv'))
                rules_nh = pd.read_csv(join(racine_path, 'nh_rules.csv'))

                rs_dict['Sirus'] += [make_rs_from_r(rules_sirus, features.to_list(),
                                                    X_train.min(axis=0), X_train.max(axis=0))]
                rs_dict['NH'] += [make_rs_from_r(rules_nh, features.to_list(), X_train.min(axis=0),
                                                 X_train.max(axis=0))]

                tree = DecisionTreeRegressor(max_leaf_nodes=10)
                tree.fit(X_train, y_train)

                rs_dict['DT'] += [extract_rules_from_tree(tree, xmins=X_train.min(axis=0), xmaxs=X_train.max(axis=0),
                                                          features_names=features, get_leaf=True)]

                rule_fit = rulefit.RuleFit(tree_size=tree_size,
                                           max_rules=max_rules)
                rule_fit.fit(X_train, y_train)

                # ## RuleFit rules part
                rules = rule_fit.get_rules()
                rules = rules[rules.coef != 0].sort_values(by="support")
                rules = rules.loc[rules['type'] == 'rule']

                rs_dict['RuleFit'] += [extract_rules_rulefit(rules, features, X_train.min(axis=0),
                                                             X_train.max(axis=0))]

                # ## Covering Algorithm RandomForest
                ca_rf = CA.CA(alpha=alpha, gamma=gamma,
                              tree_size=tree_size,
                              max_rules=max_rules,
                              generator_func=RandomForestRegressor,
                              lmax=lmax)
                ca_rf.fit(xs=sub_x, y=sub_y, features=features)
                rs_dict['CA_RF'] += [ca_rf.selected_rs]

                # ## Covering Algorithm GradientBoosting
                ca_gb = CA.CA(alpha=alpha, gamma=gamma,
                              tree_size=tree_size,
                              max_rules=max_rules,
                              generator_func=GradientBoostingRegressor,
                              lmax=lmax)
                ca_gb.fit(xs=sub_x, y=sub_y, features=features)
                rs_dict['CA_GB'] += [ca_gb.selected_rs]

                # # ## RIPE
                # ripe = RIPE.Learning(nb_bucket=q, cp=lmax, intermax=gamma)
                # ripe.fit(sub_x, sub_y)
                # rs_dict['RIPE'] += [ripe.selected_rs]

            if simu == 0:
                res_dict['DT'] = [[predictivity(pred_tree, y_test, deno_mse),
                                   q_stability(rs_dict['DT'][0], rs_dict['DT'][1],  X_train,
                                               q=q, bins_dict=bins_dict),
                                   simp[0]]]
                res_dict['RuleFit'] = [[predictivity(pred_rulefit, y_test, deno_mse),
                                        q_stability(rs_dict['RuleFit'][0], rs_dict['RuleFit'][1],
                                                    X_train, q=q, bins_dict=bins_dict),
                                        simp[1]]]
                res_dict['Sirus'] = [[predictivity(pred_sirus, y_test, deno_mse),
                                      q_stability(rs_dict['Sirus'][0], rs_dict['Sirus'][1],
                                                  X_train, q=q, bins_dict=bins_dict),
                                     simp[2]]]
                res_dict['NH'] = [[predictivity(pred_nh, y_test, deno_mse),
                                   q_stability(rs_dict['NH'][0], rs_dict['NH'][1],
                                               X_train, q=q, bins_dict=bins_dict),
                                   simp[3]]]
                res_dict['CA_RF'] = [[predictivity(pred_CA_rf, y_test, deno_mse),
                                      q_stability(rs_dict['CA_RF'][0], rs_dict['CA_RF'][1],
                                                  X_train, q=q, bins_dict=bins_dict),
                                      simp[4]]]
                res_dict['CA_GB'] = [[predictivity(pred_CA_gb, y_test, deno_mse),
                                      q_stability(rs_dict['CA_GB'][0], rs_dict['CA_GB'][1],
                                                  X_train, q=q, bins_dict=bins_dict),
                                      simp[5]]]
                # res_dict['RIPE'] = [[predictivity(pred_ripe, y_test, deno_mse),
                #                      q_stability(rs_dict['RIPE'][0], rs_dict['RIPE'][1],
                #                                  X_train, q=None, bins_dict=bins_dict),
                #                      simp[6]]]

            else:
                res_dict['DT'] = np.append(res_dict['DT'],
                                           [[predictivity(pred_tree, y_test, deno_mse),
                                             q_stability(rs_dict['DT'][0], rs_dict['DT'][1],
                                                         X_train, q=q, bins_dict=bins_dict),
                                             simp[0]]], axis=0)
                res_dict['RuleFit'] = np.append(res_dict['RuleFit'],
                                                [[predictivity(pred_rulefit, y_test, deno_mse),
                                                  q_stability(rs_dict['RuleFit'][0],
                                                              rs_dict['RuleFit'][1],
                                                              X_train, q=q, bins_dict=bins_dict),
                                                  simp[1]]], axis=0)
                res_dict['Sirus'] = np.append(res_dict['Sirus'],
                                              [[predictivity(pred_sirus, y_test, deno_mse),
                                                q_stability(rs_dict['Sirus'][0],
                                                            rs_dict['Sirus'][1],
                                                            X_train, q=q, bins_dict=bins_dict),
                                                simp[2]]], axis=0)
                res_dict['NH'] = np.append(res_dict['NH'],
                                           [[predictivity(pred_nh, y_test, deno_mse),
                                             q_stability(rs_dict['NH'][0], rs_dict['NH'][1],
                                                         X_train, q=q, bins_dict=bins_dict),
                                             simp[3]]], axis=0)
                res_dict['CA_RF'] = np.append(res_dict['CA_RF'],
                                              [[predictivity(pred_CA_rf, y_test, deno_mse),
                                                q_stability(rs_dict['CA_RF'][0],
                                                            rs_dict['CA_RF'][1],
                                                            X_train, q=q, bins_dict=bins_dict),
                                                simp[4]]], axis=0)
                res_dict['CA_GB'] = np.append(res_dict['CA_GB'],
                                              [[predictivity(pred_CA_gb, y_test, deno_mse),
                                                q_stability(rs_dict['CA_GB'][0],
                                                            rs_dict['CA_GB'][1],
                                                            X_train, q=q, bins_dict=bins_dict),
                                                simp[5]]], axis=0)
                # res_dict['RIPE'] = np.append(res_dict['RIPE'],
                #                            [[predictivity(pred_ripe, y_test, deno_mse),
                #                              q_stability(rs_dict['RIPE'][0], rs_dict['RIPE'][1],
                #                                          X_train, q=None, bins_dict=bins_dict),
                #                              simp[6]]], axis=0)
            simu += 1
        # ## Results.
        print('Predictivity score')
        print('----------------------')
        print('Decision tree predicitivty score:', np.mean(res_dict['DT'][:, 0]))
        print('Covering Algorithm RF predicitivty score:', np.mean(res_dict['CA_RF'][:, 0]))
        print('Covering Algorithm GB predicitivty score:', np.mean(res_dict['CA_GB'][:, 0]))
        print('RuleFit predicitivty score:', np.mean(res_dict['RuleFit'][:, 0]))
        print('SIRUS predicitivty score:', np.mean(res_dict['Sirus'][:, 0]))
        print('NodeHarvest predicitivty score:', np.mean(res_dict['NH'][:, 0]))
        # print('RIPE predicitivty score:', np.mean(res_dict['RIPE'][:, 0]))
        print('----------------------')
        print('Decision tree predicitivty score:', np.std(res_dict['DT'][:, 0]))
        print('Covering Algorithm RF predicitivty score:', np.std(res_dict['CA_RF'][:, 0]))
        print('Covering Algorithm GB predicitivty score:', np.std(res_dict['CA_GB'][:, 0]))
        print('RuleFit predicitivty score:', np.std(res_dict['RuleFit'][:, 0]))
        print('SIRUS predicitivty score:', np.std(res_dict['Sirus'][:, 0]))
        print('NodeHarvest predicitivty score:', np.std(res_dict['NH'][:, 0]))
        # print('RIPE predicitivty score:', np.std(res_dict['RIPE'][:, 0]))
        print('')
        print('q-Stability score')
        print('----------------------')
        print('Decision tree q-Stability score:', np.mean(res_dict['DT'][:, 1]))
        print('Covering Algorithm RF q-Stability score:', np.mean(res_dict['CA_RF'][:, 1]))
        print('Covering Algorithm GB q-Stability score:', np.mean(res_dict['CA_GB'][:, 1]))
        print('RuleFit q-Stability score:', np.mean(res_dict['RuleFit'][:, 1]))
        print('SIRUS q-Stability score:', np.mean(res_dict['Sirus'][:, 1]))
        print('NodeHarvest q-Stability score:', np.mean(res_dict['NH'][:, 1]))
        # print('RIPE q-Stability score:', np.mean(res_dict['RIPE'][:, 1]))
        print('----------------------')
        print('Decision tree q-Stability score:', np.std(res_dict['DT'][:, 1]))
        print('Covering Algorithm RF q-Stability score:', np.std(res_dict['CA_RF'][:, 1]))
        print('Covering Algorithm GB q-Stability score:', np.std(res_dict['CA_GB'][:, 1]))
        print('RuleFit q-Stability score:', np.std(res_dict['RuleFit'][:, 1]))
        print('SIRUS q-Stability score:', np.std(res_dict['Sirus'][:, 1]))
        print('NodeHarvest q-Stability score:', np.std(res_dict['NH'][:, 1]))
        # print('RIPE q-Stability score:', np.std(res_dict['RIPE'][:, 1]))
        print('')
        print('Simplicity score')
        print('----------------------')
        print('Decision tree Simplicity score:', np.mean(res_dict['DT'][:, 2]))
        print('Covering Algorithm RF Simplicity score:', np.mean(res_dict['CA_RF'][:, 2]))
        print('Covering Algorithm GB Simplicity score:', np.mean(res_dict['CA_GB'][:, 2]))
        print('RuleFit Simplicity score:', np.mean(res_dict['RuleFit'][:, 2]))
        print('SIRUS Simplicity score:', np.mean(res_dict['Sirus'][:, 2]))
        print('NodeHarvest Simplicity score:', np.mean(res_dict['NH'][:, 2]))
        # print('RIPE Simplicity score:', np.mean(res_dict['RIPE'][:, 2]))
        print('----------------------')
        print('Decision tree Simplicity score:', np.std(res_dict['DT'][:, 2]))
        print('Covering Algorithm RF Simplicity score:', np.std(res_dict['CA_RF'][:, 2]))
        print('Covering Algorithm GB Simplicity score:', np.std(res_dict['CA_GB'][:, 2]))
        print('RuleFit Simplicity score:', np.std(res_dict['RuleFit'][:, 2]))
        print('SIRUS Simplicity score:', np.std(res_dict['Sirus'][:, 2]))
        print('NodeHarvest Simplicity score:', np.std(res_dict['NH'][:, 2]))
        # print('RIPE Simplicity score:', np.std(res_dict['RIPE'][:, 2]))
