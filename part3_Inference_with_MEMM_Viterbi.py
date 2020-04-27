"""## Part 3 - Inference with MEMM-Viterbi"""
import numpy as np
import time

from scipy import sparse
from scipy.stats import mode


def tri_mat_to_probs(tri_mat, v):
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
    mat = np.sum(v[tri_mat], axis=3)
    mat = np.exp(mat)
    sum_exp = np.sum(mat, axis=2).reshape((mat.shape[0], mat.shape[1], 1))
    mat /= sum_exp
    return mat


def memm_viterbi(num_h, tri_mat_gen, v, time_run=False, iprint=500):
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
    if len(v.shape) == 1:
        mat = np.sum(v[feature_mat], axis=2)
        mat = np.exp(mat)
        sum_exp = np.sum(mat, axis=1).reshape((mat.shape[0], 1))
        mat /= sum_exp
    else:
        mat = np.sum(v[:, feature_mat], axis=3)
        mat = np.exp(mat)
        sum_exp = np.sum(mat, axis=2).reshape((mat.shape[0], 1))
        mat /= sum_exp
        mat = np.mean(mat, axis=0)
    beam_width, num_c = mat.shape
    rows = np.tile(num_p * pptags + ptags, (num_c, 1))
    rows = np.transpose(rows).reshape(-1)
    cols = np.tile(np.arange(num_c), beam_width)
    res = sparse.csr_matrix((mat.reshape(-1), (rows, cols)), (num_pp * num_p, num_c))
    return res


def memm_viterbi_beam_search(num_h, mat_gen, v, beam_width, time_run=False, iprint=500):
    tags_infer = []
    start = True
    v = np.append(v, 0)
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


def compute_accuracy(true_tags, tri_mat_gen, v, time_run=False, iprint=500):
    tags_infer = memm_viterbi(len(true_tags), tri_mat_gen, v, time_run=time_run, iprint=iprint)
    return np.sum(true_tags == tags_infer)/len(true_tags)


def compute_accuracy_beam(true_tags, mat_gen, v, beam_width, time_run=False, iprint=500):
    tags_infer = memm_viterbi_beam_search(len(true_tags), mat_gen, v, beam_width, time_run=time_run, iprint=iprint)
    return np.sum(true_tags == tags_infer)/len(true_tags)


def compute_accuracy_beam_with_hard_vote(true_tags, mat_gen, v, beam_width, time_run=False, iprint=None,
                                         first_weight=1):
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


# def compute_accuracy_beam(true_tags, mat_gen, v, beam_width1, beam_width2, time_run=False, iprint=500):
#     tags_infer1 = memm_viterbi_beam_search(18, mat_gen, v, beam_width1, time_run=time_run, iprint=iprint)
#     tags_infer2 = memm_viterbi_beam_search(18, mat_gen, v, beam_width2, time_run=time_run, iprint=iprint)
#     print(np.all(tags_infer1==tags_infer2))
#     return np.sum(true_tags[:18] == tags_infer1)/18
