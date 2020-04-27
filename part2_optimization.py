import numpy as np
import pickle
from scipy import sparse
from scipy.optimize import fmin_l_bfgs_b
import time


def feature_list_to_sparse_matrix(feature_list, return_dims=True):
    num_h = len(feature_list)
    num_t = len(feature_list[0])
    row = []
    col = []
    for h in range(num_h):
        for t in range(num_t):
            col_ind = feature_list[h][t]
            row_ind = h * num_t + t
            row.extend(len(col_ind) * [row_ind])
            col += col_ind
    row = np.array(row)
    col = np.array(col)
    mat = sparse.csr_matrix((np.ones(len(row)), (row, col)))
    if return_dims:
        num_f = mat.shape[1]
        return mat, num_h, num_t, num_f


def calc_objective_per_iter(w_i, feature_mat: sparse.csr_matrix, empirical_counts, num_h, true_tags, alpha, l1=False):
    """
        Calculate max entropy likelihood for an iterative optimization method
        :param alpha: the regularization coefficient
        :param num_h: number of histories in the training data
        :param empirical_counts: pre computed empirical_counts
        :param w_i: weights vector in iteration i


            The function returns the Max Entropy likelihood (objective) and the objective gradient
    """
    scores = feature_mat.dot(w_i)
    scores = scores.reshape((num_h, -1))
    exp_scores = np.exp(scores)
    sum_exp = np.sum(exp_scores, axis=1)
    probs = exp_scores/sum_exp.reshape((num_h, 1))
    expected_counts = feature_mat.transpose().dot(probs.reshape(-1)).reshape(-1)
    if l1:
        likelihood = np.sum(scores[np.arange(num_h), true_tags]) - np.log(sum_exp).sum() - alpha * np.abs(w_i).sum()
        grad = empirical_counts - expected_counts - alpha * np.sign(w_i)
    else:
        likelihood = np.sum(scores[np.arange(num_h), true_tags]) - np.log(sum_exp).sum() - (alpha/2) * (w_i**2).sum()
        grad = empirical_counts - expected_counts - alpha*w_i
    return (-1) * likelihood, (-1) * grad


def train_from_list(feature_list, true_tags, alpha, weights_path=None, time_run=False, feature_select=None, l1=False):
    if time_run:
        t1 = time.time()
    feature_mat, num_h, num_t, num_f = feature_list_to_sparse_matrix(feature_list)
    if time_run:
        t2 = time.time()
        print('Part 2 format conversion time:', t2 - t1)
    num_fs = num_f
    if feature_select is not None:
        feature_mat = feature_mat[:, feature_select]
        num_fs = np.sum(feature_select)
    true_tags_history = num_t * np.arange(num_h) + true_tags
    empirical_counts = np.asarray(feature_mat[true_tags_history].sum(axis=0)).reshape(-1)
    args = (feature_mat, empirical_counts, num_h, true_tags, alpha, l1)
    w_0 = np.zeros(num_fs, dtype=np.float32)
    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
    weights = np.zeros(num_f)
    weights[feature_select] = optimal_params[0]
    if time_run:
        t3 = time.time()
        print('Total part 2 run time:', t3 - t1)
    if weights_path is not None:
        with open(weights_path, 'wb') as f:
            pickle.dump(optimal_params, f)
    return weights


def feature_selector(feature_list, true_tags, alpha=0., maxiter=10, q=0.2):
    feature_mat, num_h, num_t, num_f = feature_list_to_sparse_matrix(feature_list)
    true_tags_history = num_t * np.arange(num_h) + true_tags
    empirical_counts = np.asarray(feature_mat[true_tags_history].sum(axis=0)).reshape(-1)
    args = (feature_mat, empirical_counts, num_h, true_tags, alpha)
    w_0 = np.zeros(num_f, dtype=np.float32)
    optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=maxiter, iprint=50)
    weights = np.abs(optimal_params[0])
    threshold = np.quantile(weights, q)
    return weights >= threshold


def train_ensemble_from_list(feature_list, true_tags, alpha, num_estimators=5, weights_path=None, time_run=False,
                             max_samples=1., max_features=1., bootstrap_samples=True, stack=False, maxiter=1000):
    if time_run:
        t1 = time.time()
    orig_feature_mat, num_h, num_t, num_f = feature_list_to_sparse_matrix(feature_list)
    if stack:
        v = np.zeros((num_estimators, num_f), dtype=np.float32)
    else:
        v = np.zeros(num_f, dtype=np.float32)
    for i in range(num_estimators):
        num_fs = int(max_features * num_f)
        num_hs = int(max_samples * num_h)
        if max_features != 1:
            sampled_features = np.random.permutation(num_f)[: num_fs]
            feature_mat = orig_feature_mat[:, sampled_features]
        else:
            feature_mat = orig_feature_mat
        if bootstrap_samples:
            sample = np.random.randint(num_h, size=num_hs)
        else:
            sample = np.random.permutation(num_h)[: num_hs]
        sample_true_tags = true_tags[sample]
        ht_sample = np.tile(num_t * sample, (num_t, 1))
        ht_sample += np.arange(num_t).reshape(-1, 1)
        ht_sample = np.transpose(ht_sample).reshape(-1)
        sample_feature_mat = feature_mat[ht_sample]
        true_tags_history = num_t * np.arange(num_hs) + true_tags[sample]
        empirical_counts = np.asarray(sample_feature_mat[true_tags_history].sum(axis=0)).reshape(-1)
        args = (sample_feature_mat, empirical_counts, num_hs, sample_true_tags, alpha)
        w_0 = np.zeros(num_fs, dtype=np.float32)
        optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=maxiter, iprint=50)
        if stack:
            if max_features != 1:
                v[i, sampled_features] = optimal_params[0]
            else:
                v[i] = optimal_params[0]
        else:
            if max_features != 1:
                v[sampled_features] += optimal_params[0]
            else:
                v += optimal_params[0]
    if not stack:
        v /= num_estimators
        v *= 1/max_features
    if time_run:
        t2 = time.time()
        print('Total part 2 run time:', t2 - t1)
    if weights_path is not None:
        with open(weights_path, 'wb') as f:
            pickle.dump(v, f)
    return v

'''
"""Now lets run the code untill we get the optimized weights."""

from scipy.optimize import fmin_l_bfgs_b

# Statistics
statistics = feature_statistics_class()
statistics.get_word_tag_pairs(train_path)

# feature2id
feature2id = feature2id_class(statistics, threshold)
feature2id.get_word_tag_pairs(train_path)

# define 'args', that holds the arguments arg_1, arg_2, ... for 'calc_objective_per_iter'
args = (arg_1, arg_2, ...)
w_0 = np.zeros(n_total_features, dtype=np.float32)
optimal_params = fmin_l_bfgs_b(func=calc_objective_per_iter, x0=w_0, args=args, maxiter=1000, iprint=50)
weights = optimal_params[0]

# Now you can save weights using pickle.dump() - 'weights_path' specifies where the weight file will be saved.
# IMPORTANT - we expect to recieve weights in 'pickle' format, don't use any other format!!
weights_path = 'your_path_to_weights_dir/trained_weights_data_i.pkl'  # i identifies which dataset this is trained on
with open(weights_path, 'wb') as f:
    pickle.dump(optimal_params, f)

#### In order to load pre-trained weights, just use the next code: ####
#                                                                     #
# with open(weights_path, 'rb') as f:                                 #
#   optimal_params = pickle.load(f)                                   #
# pre_trained_weights = optimal_params[0]                             #
#                                                                     #
#######################################################################
'''

