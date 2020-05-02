"""## Part 3 - Inference with MEMM-Viterbi"""
import numpy as np
import time

from scipy import sparse
from scipy.stats import mode
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


def tri_mat_to_probs(tri_mat, v):
    """
    coverts the tri_mat with pptag, ptag, ctag entries that stores the satisfied features for
    the history, pptag, ptag, ctag to probabilty array according to weights v (with the same entries)
    """
    num_ppt = len(tri_mat)
    num_pt = len(tri_mat[0])
    num_ct = len(tri_mat[0][0])
    mat = np.empty((num_ppt, num_pt, num_ct))
    for ppt in range(num_ppt):
        for pt in range(num_pt):
            for ct in range(num_ct):
                mat[ppt][pt][ct] = v[tri_mat[ppt][pt][ct]].sum()
    mat = np.exp(mat)
    sum_exp = np.sum(mat, axis=2).reshape((num_ppt, num_pt, 1))
    mat /= sum_exp
    return mat


def vectorized_tri_mat_to_probs(tri_mat, v):
    """coverts the tri_mat array with pptag, ptag, ctag entries that stores the satisfied features for each
    feature class or -1 if non is satisfied (the format help for vectorized compution) to probability
    array with the same entries.
    assumes that v[-1]==0"""
    mat = np.sum(v[tri_mat], axis=3)
    mat = np.exp(mat)
    sum_exp = np.sum(mat, axis=2).reshape((mat.shape[0], mat.shape[1], 1))
    mat /= sum_exp
    return mat


def memm_viterbi(num_h, tri_mat_gen, v, time_run=False, iprint=500):
    """
    infer tags using the viterbi algorithm
    :param num_h: number of histories
    :param tri_mat_gen: generating array with pptag, ptag, ctag entries in the format for vectorized_tri_mat_to_probs
    (as it proves to by faster) given history number, also returns the sentens the history is in and if the history is
    of the last word of the sentence.
    :param v: the trained weights
    :param time_run: if true, times the run
    :param iprint: if not None, print after every iprint iteration
    :return: array of the infefered tags (as ints)
    """
    tags_infer = []
    start = True
    v = np.append(v, 0)
    if time_run:
        t_mat_gen = 0
        t_prob = 0
        t_veterbi = 0
    k = 0
    for h in range(num_h):
        if iprint is not None:
            if h % iprint == 0:
                print('iter:', h)
        if time_run:
            t0 = time.time()
        tri_mat, _, end = tri_mat_gen(h)
        if time_run:
            t1 = time.time()
            t_mat_gen += t1-t0
        q = vectorized_tri_mat_to_probs(tri_mat, v)
        if time_run:
            t2 = time.time()
            t_prob += t2 - t1
        if start:
            if end:
                tags_infer.append(np.argmax(q[0][0]))
                k = 0
                continue
            pi = [q.reshape(q.shape[1:])]
            bp = [0]
            k = 1
            start = False
            continue
        pi_prev = pi[k-1].reshape(pi[k-1].shape[0], -1, 1)
        probs = pi_prev * q
        curr_bp = np.argmax(probs, axis=0)
        bp.append(curr_bp)
        i, j = np.ogrid[: probs.shape[1], : probs.shape[2]]
        pi.append(probs[curr_bp, i, j])
        k += 1
        if end:
            len_s = k
            curr_tags = len_s * [0]
            curr_tags[len_s-2], curr_tags[len_s-1] = np.unravel_index(np.argmax(pi[len_s-1], axis=None), pi[len_s-1].shape)
            for k in range(len_s - 3, -1, -1):
                curr_tags[k] = bp[k+2][curr_tags[k+1], curr_tags[k+2]]
            tags_infer.extend(curr_tags)
            k = 0
            start = True
        if time_run:
            t_veterbi += time.time() - t2
    if time_run:
        print('time feature matrix compution:', t_mat_gen)
        print('time probability compution:', t_prob)
        print('time veterbi:', t_veterbi)
        print('total:', t_mat_gen + t_prob + t_veterbi)
    return np.array(tags_infer)


def vectorized_mat_to_probs_for_beam_search(feature_mat, v, pptags, ptags, num_pp, num_p):
    """
    :param feature_mat: array with entry for feature request number storing the feature satisfied in the format
    described in vectorized_tri_mat_to_probs
    :param v: the weight of the model assuming v[-1]==0 if v is 2 dimensional each row of v should be weight for
    differnt models (v dimension should not be higher than 2)
    :param pptags: array of pptags requested such that pptag[i] is the i'th pptag requested
    :param ptags: array of ptags requested such that ptag[i] is the i'th ptag requested
    :param num_pp: total number of tag that can come before the history before the current history (1 at the start
    or second word of the sentence)
    :param num_p: total number of tag that can come before the current history (1 at the start of the sentence)
    :return: scipy csr sparse matrix where the cell in the num_p * t + u and th column c is the probality of tag c
    given ptag u and pptag t and the history for which feature_mat was generated if the state t, u was requested
    or 0 otherwise. (if v is 2 dimensional the result are averaged)
    """
    if len(v.shape) == 1:
        mat = np.sum(v[feature_mat], axis=2)
        mat = np.exp(mat)
        sum_exp = np.sum(mat, axis=1).reshape((mat.shape[0], 1))
        mat /= sum_exp
    else:
        mat = np.sum(v[:, feature_mat], axis=3)
        mat = np.exp(mat)
        sum_exp = np.sum(mat, axis=2).reshape((mat.shape[0], -1, 1))
        mat /= sum_exp
        mat = np.mean(mat, axis=0)
    beam_width, num_c = mat.shape
    rows = np.tile(num_p * pptags + ptags, (num_c, 1))
    rows = np.transpose(rows).reshape(-1)
    cols = np.tile(np.arange(num_c), beam_width)
    res = sparse.csr_matrix((mat.reshape(-1), (rows, cols)), (num_pp * num_p, num_c))
    return res


def memm_viterbi_beam_search(num_h, mat_gen, v, beam_width, time_run=False, iprint=500):
    """
    infer tags using the viterbi algorithm with the beam search heuristic. the implementation uses scipy sparse
    matrix to store pi, q and the bp.
    :param num_h: number of histories
    :param mat_gen: generating array that store the feature satisfied (in the format
    described in vectorized_tri_mat_to_probs) with entry for request number given the history number for which
    the request is acted iterator of pptag, ptag pair requested and the number of requests
    :param v: v the weight of the model (if two dimensional store in each row is weight for different model
    and the result are aggregated in vectorized_mat_to_probs_for_beam_search)
    :param beam_width: the beam width
    :param time_run: if true, times the run
    :param iprint: if not None, print after every iprint iteration
    :return: array of the infefered tags (as ints)
    """
    tags_infer = []
    start = True
    if len(v.shape) == 1:
        v = np.append(v, 0)
    else:
        v_with_zeros = np.zeros((v.shape[0], v.shape[1]+1))
        v_with_zeros[:, : -1] = v
        v = v_with_zeros
    k = 0
    pi = []
    bp = []
    selected_tags = (np.zeros(1, dtype=int), np.zeros(1, dtype=int))
    curr_beam = 1
    ##
    if time_run:
        t_mat_gen = 0
        t_prob = 0
        t_veterbi = 0
    ##

    for h in range(num_h):
        if iprint is not None:
            if h % iprint == 0:
                print('iter:', h)
        if time_run:
            t0 = time.time()
        mat, _, end, num_pp, num_p = mat_gen(h, zip(*selected_tags), curr_beam)
        if time_run:
            t1 = time.time()
            t_mat_gen += t1 - t0
        num_c = mat.shape[1]
        q = vectorized_mat_to_probs_for_beam_search(mat, v, selected_tags[0], selected_tags[1], num_pp, num_p)

        if time_run:
            t2 = time.time()
            t_prob += t2 - t1

        if start:
            if end:
                tags_infer.append(np.argmax(q))
                continue
            pi = []
            curr_pi = q.tocoo(copy=True).reshape(-1, 1)
            bp = [0]
        else:
            probs = q.multiply(pi[k-1])
            probs = probs.reshape(num_pp, -1)
            curr_pi = probs.max(axis=0).reshape(-1, 1)
            probs = probs.tocsc()
        nnz_pi = curr_pi.data.shape[0]
        curr_beam = min(nnz_pi, beam_width)
        selected_data_ind = curr_pi.data.argpartition(nnz_pi - curr_beam)[-curr_beam:]
        selected_ind = curr_pi.row[selected_data_ind]
        selected_tags = np.unravel_index(selected_ind, (num_p, num_c))
        curr_pi = curr_pi.tocsr()
        pi.append(curr_pi)
        if not start:
            bp_vals = probs[:, selected_ind].argmax(axis=0).A1
            bp.append(sparse.csr_matrix((bp_vals, selected_tags)))
        else:
            start = False
        k += 1
        if end:
            len_s = k
            curr_tags = len_s * [0]
            curr_tags[len_s-2], curr_tags[len_s-1] = np.unravel_index(pi[len_s-1].argmax(), (num_p, num_c))
            for k in range(len_s - 3, -1, -1):
                curr_tags[k] = bp[k+2][curr_tags[k+1], curr_tags[k+2]]
            tags_infer.extend(curr_tags)
            k = 0
            selected_tags = (np.zeros(1, dtype=int), np.zeros(1, dtype=int))
            curr_beam = 1
            start = True
        if time_run:
            t_veterbi += time.time() - t2
    if time_run:
        print('time feature matrix compution:', t_mat_gen)
        print('time probability compution:', t_prob)
        print('time veterbi:', t_veterbi)
        print('total:', t_mat_gen + t_prob + t_veterbi)
    return np.array(tags_infer)


def compute_accuracy(true_tags, tri_mat_gen, v, time_run=False, iprint=1000):
    """computing the accuracy of the model using the veterbi algorithm (for the file tri_mat_gen generates the
    arrays for)
    true_tags is an array for the actual tags (as ints)"""
    tags_infer = memm_viterbi(len(true_tags), tri_mat_gen, v, time_run=time_run, iprint=iprint)
    return np.sum(true_tags == tags_infer)/len(true_tags)


def compute_accuracy_beam(true_tags, mat_gen, v, beam_width, time_run=False, iprint=1000):
    """computing the accuracy of the model using the veterbi algorithm with beam search with beam width beam_width
    (for the file mat_gen generates the arrays for)
    true_tags is an array for the actual tags (as ints)"""
    tags_infer = memm_viterbi_beam_search(len(true_tags), mat_gen, v, beam_width, time_run=time_run, iprint=iprint)
    return np.sum(true_tags == tags_infer)/len(true_tags), tags_infer


def infer_tags(num_h, mat_gen, v, beam_width, tag_list, time_run=False, iprint=None):
    """infer the tags in string format using veterbi algorithm with beam search (conversion to string done by using the
    argument tag_list which is a list such that tag_list[i] == the string representation of the tag encoded as i"""
    tags_infer_ind = memm_viterbi_beam_search(num_h, mat_gen, v, beam_width, time_run=time_run, iprint=iprint)
    tags_infer = []
    for ind in tags_infer_ind:
        tags_infer.append(tag_list[ind])
    return tags_infer


def compute_accuracy_beam_with_hard_vote(true_tags, mat_gen, v, beam_width, time_run=False, iprint=None,
                                         first_weight=1):
    """computing the accuracy of hard vote done between multiple models using hrad vote (chosing the tag most
    model predicted) where the veterbi algorithm with beam search used for infernce for each model.
    assumes that v is 2 dimensional and each row is the weight of different estimator.
    first weight is the weight the first model is given (in the case there is one model that is more trusted)"""
    num_estimators = v.shape[0]
    num_h = len(true_tags)
    tags_infer_votes = np.empty((num_estimators + first_weight - 1, num_h), dtype=int)
    for i in range(num_estimators):
        tags_infer_votes[i] = memm_viterbi_beam_search(num_h, mat_gen, v[i], beam_width,
                                                       time_run=time_run, iprint=iprint)
    for i in range(first_weight-1):
        tags_infer_votes[i + num_estimators] = tags_infer_votes[0]
    tags_infer = mode(tags_infer_votes)[0]
    return np.sum(true_tags == tags_infer)/len(true_tags)


def plot_confusion_matrix(true_tags, mat_gen, v, beam_width, tag_list, zero_diag=True):
    """plot the confusion matrix for the ten tag our the model is most confused for (where inference is done using
    beam search) if the zero_diag argument the diagonal is zeroed to allow the errors to be seen"""
    tags_infer = memm_viterbi_beam_search(len(true_tags), mat_gen, v, beam_width)
    tags_infer_df = pd.Series(tags_infer, name='Predicted')
    true_tags_df = pd.Series(true_tags, name='Actual')
    confusion_matrix = pd.crosstab(tags_infer_df, true_tags_df)
    seen_tags_ind = np.copy(confusion_matrix.index.values)
    diag = [0 for _ in range(max(seen_tags_ind) + 1)]
    for i in confusion_matrix.index.values:
        diag[i] = confusion_matrix[i][i]
        confusion_matrix[i][i] = 0
    confusion_matrix.rename(columns=lambda s: tag_list[int(s)], index=lambda s: tag_list[int(s)], inplace=True)
    confusion_order = confusion_matrix.sum(axis=0).sort_values()[::-1].index
    if not zero_diag:
        for i in seen_tags_ind:
            tag = tag_list[i]
            confusion_matrix[tag][tag] = diag[i]
    confusion_matrix = confusion_matrix[confusion_order[: 10]]
    confusion_matrix = confusion_matrix.reindex(confusion_order.rename('Predicted'), copy=False, fill_value=0)
    plt.figure(figsize=(20, 10))
    seaborn.heatmap(confusion_matrix ,annot=True)
    plt.show()


def get_test_statistics(true_tags, mat_gen, v, beam_width, sentence_indexes, num_sample=20):
    """davids the test sentences to num_sample sample computing the accuracy on each one
    and prints the mean, min, max and confidence interval using these results"""
    num_s = len(sentence_indexes)
    tags_infer = memm_viterbi_beam_search(len(true_tags), mat_gen, v, beam_width)
    correct_predicted = (true_tags == tags_infer)
    correct_v_total = np.zeros((num_s, 2), dtype=int)
    for i, (start, end) in enumerate(sentence_indexes):
        correct_v_total[i, 0], correct_v_total[i, 1] = np.sum(correct_predicted[start: end]), end - start
    perm = np.random.permutation(num_s)
    sample_size = num_s//num_sample
    res = np.zeros(num_sample)
    for i in range(num_sample):
        sample = perm[i * sample_size: (i+1) * sample_size]
        num_correct = np.sum(correct_v_total[sample, 0])
        num_total = np.sum(correct_v_total[sample, 1])
        res[i] = num_correct / num_total
    acc = np.mean(res)
    sd = np.std(res)
    interval_radius = 1.96 * sd / np.sqrt(num_sample)
    print('mean accuracy:', acc)
    print('min:', np.min(res))
    print('max:', np.max(res))
    print('confidence interval:', '[', acc - interval_radius, ',', acc + interval_radius, ']')
    return res
