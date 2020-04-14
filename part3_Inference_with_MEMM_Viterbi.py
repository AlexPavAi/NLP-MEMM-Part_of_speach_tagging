"""## Part 3 - Inference with MEMM-Viterbi"""
import numpy as np
import time


def feature_list_to_probs(sentences_features_list, v):
    sentences_q_list = []
    for feature_list in sentences_features_list:
        q = []
        for tri_mat in feature_list:
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
            q.append(mat)
        sentences_q_list.append(q)
    return sentences_q_list


def memm_viterbi(sentences_q_list):
    tags_infer = []
    for q in sentences_q_list:
        num_h = len(q)
        curr_tags = [0] * num_h
        pi = [q[0].reshape(q[0].shape[1:])]
        bp = [0]
        for k in range(1, num_h):
            pi_prev = pi[k-1].reshape(pi[k-1].shape[0], -1, 1)
            probs = pi_prev * q[k]
            curr_bp = np.argmax(probs, axis=0)
            bp.append(curr_bp)
            i, j = np.ogrid[: probs.shape[1], : probs.shape[2]]
            pi.append(probs[curr_bp, i, j])
        curr_tags[num_h-2], curr_tags[num_h-1] = np.unravel_index(np.argmax(pi[num_h-1], axis=None), pi[num_h-1].shape)
        for k in range(num_h - 3, -1, -1):
            curr_tags[k] = bp[k+2][curr_tags[k+1], curr_tags[k+2]]
        tags_infer.extend(curr_tags)
    return np.array(tags_infer)


def compute_accuracy(sentences_features_list, v, true_tags, time_run=False):
    if time_run:
        t0 = time.time()
    sentences_q_list = feature_list_to_probs(sentences_features_list, v)
    if time_run:
        t1 = time.time()
        print('Part 3 Probability compution time:', t1 - t0)
    tags_infer = memm_viterbi(sentences_q_list)
    if time_run:
        t2 = time.time()
        print('viterbi\'s run time:', t2 - t1)
        print('Total part 3 run time:', t2 - t0)
    return  np.sum(true_tags == tags_infer)/len(true_tags)
