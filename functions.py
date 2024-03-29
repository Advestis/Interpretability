from typing import Union, List
import copy
import pandas as pd
import numpy as np


from rule.rule import Rule
from condition.hyperrectanglecondition import HyperrectangleCondition
from ruleset.ruleset import RuleSet

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import _tree


def predictivity(y_hat, y, deno):
    assert len(y_hat) == len(y)
    return 1 - np.mean((y - y_hat) ** 2) / deno


def predictivity_classif(y_hat, y):
    assert len(y_hat) == len(y)
    return sum([x == z for x, z in zip(y_hat, y)]) / len(y)


def simplicity(rs: Union[RuleSet, List[Rule]]) -> int:
    return sum(map(lambda r: len(r), rs))


def find_bins(x, nb_bucket):
    """
    Function used to find the bins to discretize xcol in nb_bucket modalities

    Parameters
    ----------
    x : {Series type}
           Serie to discretize

    nb_bucket : {int type}
                Number of modalities

    Return
    ------
    bins : {ndarray type}
           The bins for disretization (result from numpy percentile function)
    """
    # Find the bins for nb_bucket
    x = x.astype('float')
    q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
    bins = np.array([np.nanpercentile(x, i) for i in q_list])

    if bins.min() != 0:
        test_bins = bins / bins.min()
    else:
        test_bins = bins

    # Test if we have same bins...
    while len(set(test_bins.round(5))) != len(bins):
        # Try to decrease the number of bucket to have unique bins
        nb_bucket -= 1
        q_list = np.arange(100.0 / nb_bucket, 100.0, 100.0 / nb_bucket)
        bins = np.array([np.nanpercentile(x, i) for i in q_list])
        if bins.min() != 0:
            test_bins = bins / bins.min()
        else:
            test_bins = bins

    return bins


def bound_to_bins(rule, q, X, bins_dict=None):
    rcond = rule.condition
    bmin_bins = []
    bmax_bins = []
    geq_min = True
    leq_min = True
    not_nan = True
    for k in range(len(rule)):
        var_index = rcond.features_indexes[k]
        xcol = X[:, var_index]
        if bins_dict is None:
            var_bins = find_bins(xcol, q)
        else:
            var_bins = bins_dict[rcond.features_names[k]]

        bmin_bins += list(np.digitize(rcond.bmins[k:k + 1], var_bins))
        bmax_bins += list(np.digitize(rcond.bmaxs[k:k + 1], var_bins))

        xcol = np.digitize(xcol, bins=var_bins)

        x_temp = [bmin_bins[k] - 1 if x != x else x for x in xcol]
        geq_min &= np.greater_equal(x_temp, bmin_bins[k])

        x_temp = [bmax_bins[k] + 1 if x != x else x for x in xcol]
        leq_min &= np.less_equal(x_temp, bmax_bins[k])

        not_nan &= np.isfinite(xcol)

    new_cond = HyperrectangleCondition(rcond.features_indexes, bmin_bins, bmax_bins, rcond.features_names)
    new_rule = Rule(new_cond)
    activation_vector = 1 * (geq_min & leq_min & not_nan)

    new_rule.activation = activation_vector
    return new_rule


def q_stability(rs1, rs2, X, q=None, bins_dict=None):
    if q is not None:
        q_rs1 = RuleSet([bound_to_bins(rule, q, X, bins_dict) for rule in rs1])
        q_rs2 = RuleSet([bound_to_bins(rule, q, X, bins_dict) for rule in rs2])
    else:
        q_rs1 = rs1
        q_rs2 = rs2
    return 2*len(set(q_rs1).intersection(q_rs2)) / (len(q_rs1) + len(q_rs2))


def extract_rules_rulefit(rules: pd.DataFrame,
                          features: List[str],
                          bmin_list: List[float],
                          bmax_list: List[float]) -> List[Rule]:
    rule_list = []

    for rule in rules['rule'].values:
        if '&' in rule:
            rule_split = rule.split(' & ')
        else:
            rule_split = [rule]

        features_name = []
        features_index = []
        bmin = []
        bmax = []

        for sub_rule in rule_split:
            sub_rule = sub_rule.replace('=', '')

            if '>' in sub_rule:
                sub_rule = sub_rule.split(' > ')
                if 'feature_' in sub_rule[0]:
                    feat_id = sub_rule[0].split('_')[-1]
                    feat_id = int(feat_id)
                    features_name += [features[feat_id]]
                else:
                    features_name += [sub_rule[0]]
                    feat_id = features.index(sub_rule[0])
                features_index += [feat_id]
                bmin += [float(sub_rule[-1])]
                bmax += [bmax_list[feat_id]]
            else:
                sub_rule = sub_rule.split(' < ')
                if 'feature_' in sub_rule[0]:
                    feat_id = sub_rule[0].split('_')[-1]
                    feat_id = int(feat_id)
                    features_name += [features[feat_id]]
                else:
                    features_name += [sub_rule[0]]
                    feat_id = features.index(sub_rule[0])
                features_index += [feat_id]
                bmax += [float(sub_rule[-1])]
                bmin += [bmin_list[feat_id]]

        new_cond = HyperrectangleCondition(features_indexes=features_index,
                                           bmins=bmin, bmaxs=bmax,
                                           features_names=features_name)
        new_rg = Rule(copy.deepcopy(new_cond))
        rule_list.append(new_rg)

    return rule_list


def make_rs_from_r(df, features_list, xmin, xmax):
    rules = df['Rules'].dropna().values
    rule_list = []
    for i in range(len(rules)):
        if 'in' in rules[i]:
            rl_i = rules[i].split(' AND ')
            cp = len(rl_i)
            conditions = [[] for _ in range(6)]

            for j in range(cp):
                feature_name = rl_i[j].split(' in ')[0]
                feature_name = feature_name.replace('.', ' ')
                feature_id = features_list.index(feature_name)
                bmin = rl_i[j].split(' in ')[1].split(';')[0].replace(" ", "")
                if bmin == '-Inf':
                    bmin = xmin[feature_id]
                else:
                    bmin = float(bmin)
                bmax = rl_i[j].split(' in ')[1].split(';')[1].replace(" ", "")
                if bmax == 'Inf':
                    bmax = xmax[feature_id]
                else:
                    bmax = float(bmax)

                conditions[0] += [feature_name]
                conditions[1] += [feature_id]
                conditions[2] += [bmin]
                conditions[3] += [bmax]
                conditions[4] += [xmin[feature_id]]
                conditions[5] += [xmax[feature_id]]

            new_cond = HyperrectangleCondition(conditions[1], conditions[2], conditions[3],
                                               conditions[0])
            rule_list.append(Rule(new_cond))

    return RuleSet(rule_list)
