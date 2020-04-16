"""## Part 3 - Inference with MEMM-Viterbi"""
import numpy as np
import time


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


def compute_accuracy(true_tags, tri_mat_gen, v, time_run=False, iprint=500):
    tags_infer = memm_viterbi(len(true_tags), tri_mat_gen, v, time_run=time_run, iprint=iprint)
    return np.sum(true_tags == tags_infer)/len(true_tags)
