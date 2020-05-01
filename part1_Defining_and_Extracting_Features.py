import os
import pickle
from collections import OrderedDict
import numpy as np
import time
from part2_optimization import train_from_list
from part3_Inference_with_MEMM_Viterbi import compute_accuracy, compute_accuracy_beam, plot_confusion_matrix, \
    get_test_statistics
import math
"""
*   Pre-training:
    1.   Preprocessing data
    2.   Features engineering
    3.   Define the objective of the model


*   During training:
    1.   Represent data as feature vectors (Token2Vector)
    2.   Optimization - We need to tune the weights of the model inorder to solve the objective

*   Inference:
    1.   Use dynamic programing (Viterbi) to tag new data based on MEMM
"""


"""## Part 1 - Defining and Extracting Features
In class we saw the importance of extracting good features for NLP task. A good feature is such that (1) appear many times in the data and (2) gives information that is relevant for the label.

### Counting feature appearances in data
We would like to include features that appear many times in the data. Hence we first count the number of appearances for each feature. \
This is done at pre-training step.
"""


class FeatureStatisticsClass:

    def __init__(self, num_dicts):
        self.n_total_features = 0  # Total number of features accumulated
        self.num_features = num_dicts
        # Init all features dictionaries
        self.array_count_dicts = []
        for i in range(0, num_dicts):
            self.array_count_dicts.append(OrderedDict())

    def get_percentiles(self, percentile):
        percentiles = []
        for curr_dict in self.array_count_dicts:
            all_values = curr_dict.values()
            all_values_np = np.fromiter(all_values, dtype=int)
            curr_percentile = np.percentile(all_values_np, percentile)
            percentiles.append(curr_percentile)
        return percentiles

    def get_word_tag_pair_count_100(self, file_path):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        curr_dict = 0
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if (cur_word, cur_tag) not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][(cur_word, cur_tag)] = 1
                    else:
                        self.array_count_dicts[curr_dict][(cur_word, cur_tag)] += 1

    # --- ADD YOURE CODE BELOW --- #
    def get_word_tag_pair_count_101(self, file_path, min_length_of_suffix, max_length_of_suffix):
        """
            Extract out of text all suffixes with length 3/tag pairs
            :param file_path: full path of the file to read
                return all suffixes with length up to 4/tag pairs with index of appearance
        """
        curr_dict = 1
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if len(cur_word) > max_length_of_suffix:
                        for suffix_length in range(- max_length_of_suffix, 1 - min_length_of_suffix):
                            i_letter_suffix = cur_word[suffix_length:]
                            if (i_letter_suffix, cur_tag) not in self.array_count_dicts[curr_dict]:
                                self.array_count_dicts[curr_dict][(i_letter_suffix, cur_tag)] = 1
                            else:
                                self.array_count_dicts[curr_dict][(i_letter_suffix, cur_tag)] += 1



    def get_word_tag_pair_count_102(self, file_path, min_length_of_prefix, max_length_of_prefix):  # currently checks only if word begins with "pre"
        """
            Extract out of text all prefixes with length 3/tag pairs
            :param file_path: full path of the file to read
                return all prefixes with length up to 4/tag pairs with index of appearance
        """
        curr_dict = 2
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if len(cur_word) > max_length_of_prefix:
                        for prefix_length in range(min_length_of_prefix, max_length_of_prefix + 1):
                            i_letter_prefix = cur_word[:prefix_length]
                            if (i_letter_prefix, cur_tag) not in self.array_count_dicts[curr_dict]:
                                self.array_count_dicts[curr_dict][(i_letter_prefix, cur_tag)] = 1
                            else:
                                self.array_count_dicts[curr_dict][(i_letter_prefix, cur_tag)] += 1

    def get_tag_threesome_count_103(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
                return all threesomes of consecutive tags with index of appearance
        """
        curr_dict = 3
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(2, len(splited_words)):  # pay attention: starting from idx 2 due to the need of having two previous tags
                    ppword, pptag = splited_words[word_idx - 2].split('_')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    three_consecutive_tags = (pptag, ptag, ctag)
                    if three_consecutive_tags not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][three_consecutive_tags] = 1
                    else:
                        self.array_count_dicts[curr_dict][three_consecutive_tags] += 1

    def get_tag_couples_count_104(self, file_path):
        """
            Extract out of couples of consecutive tags
            :param file_path: full path of the file to read
                return all couples of consecutive tags with index of appearance
        """
        curr_dict = 4
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(1, len(splited_words)): ## pay attention: starting from idx 1 due to the need of having one previous tag
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    couple_consecutive_tags = (ptag, ctag)
                    if couple_consecutive_tags not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][couple_consecutive_tags] = 1
                    else:
                        self.array_count_dicts[curr_dict][couple_consecutive_tags] += 1

    def get_tag_count_105(self, file_path):
        """
            Extract out of text all tags
            :param file_path: full path of the file to read
                return all tags with index of appearance
        """
        curr_dict = 5
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(0, len(splited_words)):
                    cword, ctag = splited_words[word_idx].split('_')
                    if ctag not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][ctag] = 1
                    else:
                        self.array_count_dicts[curr_dict][ctag] += 1

    def get_prev_word_curr_tag_pair_count_106(self, file_path):
        """
            Extract out of text all prev word/tag pairs
            :param file_path: full path of the file to read
                return all prev word/tag pairs with index of appearance
        """
        curr_dict = 6
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(1, len(splited_words)):
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    if (pword, ctag) not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][(pword, ctag)] = 1
                    else:
                        self.array_count_dicts[curr_dict][(pword, ctag)] += 1

    def get_next_word_curr_tag_pair_count_107(self, file_path):
        """
            Extract out of text all next word/tag pairs
            :param file_path: full path of the file to read
                return all next word/tag pairs with index of appearance
        """
        curr_dict = 7
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(0, len(splited_words)-1):
                    cword, ctag = splited_words[word_idx].split('_')
                    nword, ntag = splited_words[word_idx + 1].split('_')
                    if (nword, ctag) not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][(nword, ctag)] = 1
                    else:
                        self.array_count_dicts[curr_dict][(nword, ctag)] += 1

    def get_tag_threesome_count_f3(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
                return all threesomes of tag + 2 previous words
        """
        curr_dict = 8
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(2, len(
                        splited_words)):  # pay attention: starting from idx 2 due to the need of having two previous tags
                    ppword, pptag = splited_words[word_idx - 2].split('_')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    tag_and_two_previous_words = (ppword, pword, ctag)
                    if tag_and_two_previous_words not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][tag_and_two_previous_words] = 1
                    else:
                        self.array_count_dicts[curr_dict][tag_and_two_previous_words] += 1

    def get_tag_word_count_f8(self, file_path, all_words):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
            :param all_words: a list containing all different words in corpus
                return all word-tag pair, s.t. word doesn't appear in previous two words
        """
        curr_dict = 9
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(2, len(
                        splited_words)):  # pay attention: starting from idx 2 due to the need of having two previous tags
                    ppword, pptag = splited_words[word_idx - 2].split('_')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    for word in all_words:
                        if ppword != word and pword != word:
                            word_and_tag = (word, ctag)
                            if word_and_tag not in self.array_count_dicts[curr_dict]:
                                self.array_count_dicts[curr_dict][word_and_tag] = 1
                            else:
                                self.array_count_dicts[curr_dict][word_and_tag] += 1


    def get_tag_word_count_capital_letter(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
            :param all_words: a list containing all different words in corpus
                return all word-tag pair, s.t. word begins with a capital letter
        """
        curr_dict = 9
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(len(splited_words)):
                    cur_word, cur_tag = splited_words[word_idx].split('_')
                    if str(cur_word[0]).isupper():
                        if (cur_word, cur_tag) not in self.array_count_dicts[curr_dict]:
                            self.array_count_dicts[curr_dict][(cur_word, cur_tag)] = 1
                        else:
                            self.array_count_dicts[curr_dict][(cur_word, cur_tag)] += 1

    def get_tag_foursome_count(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
                return all threesomes of consecutive tags with index of appearance
        """
        curr_dict = 9
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(2, len(splited_words) - 1):  # pay attention: starting from idx 2 due to the need of having two previous tags
                    ppword, pptag = splited_words[word_idx - 2].split('_')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    nword, ntag = splited_words[word_idx + 1].split('_')
                    four_consecutive_tags = (pptag, ptag, ctag, ntag)
                    if four_consecutive_tags not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][four_consecutive_tags] = 1
                    else:
                        self.array_count_dicts[curr_dict][four_consecutive_tags] += 1


    def get_tag_threesome_count_tag_cur_word_prev_word(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
                return all threesomes of tag + 2 previous words
        """
        curr_dict = 10
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                for word_idx in range(1, len(
                        splited_words)):  # pay attention: starting from idx 1 due to the need of having one previous tags
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    cword, ctag = splited_words[word_idx].split('_')
                    tag_word_prev_word = (pword, cword, ctag)
                    if tag_word_prev_word not in self.array_count_dicts[curr_dict]:
                        self.array_count_dicts[curr_dict][tag_word_prev_word] = 1
                    else:
                        self.array_count_dicts[curr_dict][tag_word_prev_word] += 1


    def get_tag_is_first_in_sentence(self, file_path):
        """
            Extract out of threesomes of consecutive tags
            :param file_path: full path of the file to read
                return all threesomes of tag + 2 previous words
        """
        curr_dict = 11
        with open(file_path) as f:
            for line in f:
                splited_words = line.split(' ')
                del splited_words[-1]
                cword, ctag = splited_words[0].split('_') # first word, tag in sentence
                if ctag not in self.array_count_dicts[curr_dict]:
                    self.array_count_dicts[curr_dict][ctag] = 1
                else:
                    self.array_count_dicts[curr_dict][ctag] += 1
"""### Indexing features 
After getting feature statistics, each feature is given an index to represent it. We include only features that appear more times in text than the lower bound - 'threshold'
"""


class Feature2idClass:

    def __init__(self, feature_statistics, thresholds, num_feautres, min_length_of_suf_pre_fix, max_length_suf_pre_fix):
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.thresholds = thresholds  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated
        self.n_tag_pairs = 0  # Number of Word\Tag pairs features
        self.featureIDX = 0   # index for each feature
        self.min_length_of_suf_pre_fix = min_length_of_suf_pre_fix
        self.max_length_suf_pre_fix = max_length_suf_pre_fix

        self.tag_list = []
        self.tag_to_ind = {}
        self.num_feature_class = -1
        # Init all features dictionaries
        self.array_of_words_tags_dicts = []
        for i in range(0, num_feautres):
            self.array_of_words_tags_dicts.append(OrderedDict())

    def get_id_for_features_over_threshold(self, file_path, num_features, min_length_suf_pre_fix, max_length_suf_pre_fix, features_list):
        """
            Extract out of text all word/tag pairs
            :param file_path: full path of the file to read
                return all word/tag pairs with index of appearance
        """
        for i in range(0, num_features):
            with open(file_path) as f:

                for line in f:
                    splited_words = line.split(' ')
                    del splited_words[-1]

                    for word_idx in range(len(splited_words)):
                        cur_word, cur_tag = splited_words[word_idx].split('_')

                        if i == 0 and features_list[i]:
                            if ((cur_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                             and (self.feature_statistics.array_count_dicts[i][(cur_word, cur_tag)] >= self.thresholds[i]):
                                self.array_of_words_tags_dicts[i][(cur_word, cur_tag)] = self.featureIDX
                                self.featureIDX += 1
                                self.n_tag_pairs += 1

                        elif i == 1:
                            th_index_extra = 0
                            if features_list[i]:
                                if len(cur_word) > max_length_suf_pre_fix:

                                    for suffix_length in range(- max_length_suf_pre_fix, 1 - min_length_suf_pre_fix):  # suffix length in absolute value
                                        i_letter_suffix = cur_word[suffix_length:]

                                        if ((i_letter_suffix, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][(i_letter_suffix, cur_tag)] >= self.thresholds[i + th_index_extra]):
                                            self.array_of_words_tags_dicts[i][(i_letter_suffix, cur_tag)] = self.featureIDX
                                            self.featureIDX += 1
                                            self.n_tag_pairs += 1
                                        th_index_extra += 1


                        elif i == 2:
                            th_index_extra = max_length_suf_pre_fix - min_length_suf_pre_fix + 1
                            if features_list[i]:
                                if len(cur_word) > max_length_suf_pre_fix:
                                    for prefix_length in range(min_length_suf_pre_fix, max_length_suf_pre_fix + 1):
                                        i_letter_prefix = cur_word[:prefix_length]

                                        if ((i_letter_prefix, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                                and (self.feature_statistics.array_count_dicts[i][
                                                         (i_letter_prefix, cur_tag)] >= self.thresholds[i + th_index_extra]):
                                            self.array_of_words_tags_dicts[i][(i_letter_prefix, cur_tag)] = self.featureIDX
                                            self.featureIDX += 1
                                            self.n_tag_pairs += 1
                                            th_index_extra += 1

                        elif i == 3 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx >= 2:
                                ppword, pptag = splited_words[word_idx - 2].split('_')
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                three_consecutive_tags = (pptag, ptag, ctag)

                                if (three_consecutive_tags not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 three_consecutive_tags] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][three_consecutive_tags] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 4 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx >= 1:
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                couple_consecutive_tags = (ptag, ctag)

                                if (couple_consecutive_tags not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 couple_consecutive_tags] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][couple_consecutive_tags] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 5 and features_list[i]:
                                th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                                cword, ctag = splited_words[word_idx].split('_')

                                if (ctag not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][ctag] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][ctag] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 6 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx > 0:
                                prev_word, prev_tag = splited_words[word_idx - 1].split('_')
                                if ((prev_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                 and (self.feature_statistics.array_count_dicts[i][(prev_word, cur_tag)] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][(prev_word, cur_tag)] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 7 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx < len(splited_words) - 1:
                                next_word, next_tag = splited_words[word_idx + 1].split('_')
                                if ((next_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                                 and (self.feature_statistics.array_count_dicts[i][(next_word, cur_tag)] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][(next_word, cur_tag)] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 8 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx >= 2:
                                ppword, pptag = splited_words[word_idx - 2].split('_')
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                tag_and_previous_two_tags = (ppword, pword, ctag)

                                if (tag_and_previous_two_tags not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 tag_and_previous_two_tags] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][tag_and_previous_two_tags] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        # elif i == 9 and features_list[i]:
                        #     if str(cur_word[0]).isupper():
                        #         if ((cur_word, cur_tag) not in self.array_of_words_tags_dicts[i]) \
                        #                 and (self.feature_statistics.array_count_dicts[i][(cur_word, cur_tag)] >=
                        #                      self.thresholds[i]):
                        #             self.array_of_words_tags_dicts[i][(cur_word, cur_tag)] = self.featureIDX
                        #             self.featureIDX += 1
                        #             self.n_tag_pairs += 1

                        elif i == 9 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx >= 2 and word_idx < len(splited_words) - 1:
                                ppword, pptag = splited_words[word_idx - 2].split('_')
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                nword, ntag = splited_words[word_idx + 1].split('_')
                                four_consecutive_tags = (pptag, ptag, ctag, ntag)

                                if (four_consecutive_tags not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 four_consecutive_tags] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][four_consecutive_tags] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        elif i == 10 and features_list[i]:
                            th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                            if word_idx >= 1:
                                pword, ptag = splited_words[word_idx - 1].split('_')
                                cword, ctag = splited_words[word_idx].split('_')
                                tag_word_prev_word = (pword, cword, ctag)

                                if (tag_word_prev_word not in self.array_of_words_tags_dicts[i]) \
                                        and (self.feature_statistics.array_count_dicts[i][
                                                 tag_word_prev_word] >= self.thresholds[i+th_index_extra]):
                                    self.array_of_words_tags_dicts[i][tag_word_prev_word] = self.featureIDX
                                    self.featureIDX += 1
                                    self.n_tag_pairs += 1

                        # elif i == 11 and features_list[i]:
                        #     th_index_extra = 2 * (max_length_suf_pre_fix - min_length_suf_pre_fix + 1)
                        #     if word_idx == 0:
                        #         cword, ctag = splited_words[word_idx].split('_')
                        #
                        #         if (ctag not in self.array_of_words_tags_dicts[i]) \
                        #                 and (self.feature_statistics.array_count_dicts[i][
                        #                          ctag] >= self.thresholds[i+th_index_extra]):
                        #             self.array_of_words_tags_dicts[i][ctag] = self.featureIDX
                        #             self.featureIDX += 1
                        #             self.n_tag_pairs += 1

                        else:
                            pass

        self.n_total_features += self.n_tag_pairs

    # --- ADD YOURE CODE BELOW --- #


"""### Representing input data with features 
After deciding which features to use, we can represent input tokens as sparse feature vectors. This way, a token is represented with a vec with a dimension D, where D is the total amount of features. \
This is done at training step.

### History tuple
We define a tuple which hold all relevant knowledge about the current word, i.e. all that is relevant to extract features for this token.
"""


def represent_input_with_features(history, Feature2idClass, ctag_input = None, pptag_input = None, ptag_input = None):
    """
        Extract feature vector in per a given history
        :param history: touple{ppword, pptag, pword, ptag, cword, ctag, nword, ntag}
        :param Feature2idClass - in order to be able to reach easily all its methods
        :param ctag_input
        :param pptag_input
        :param ptag input
        pay attention to the order!!!
            Return a list with all features that are relevant to the given history
    """

    ppword = history[0]
    pptag = history[1]
    pword = history[2]
    ptag = history[3]
    cword = history[4]
    ctag = history[5]
    nword = history[6]
    ntag = history[7]

    if pptag_input:
        pptag = pptag_input
    if ptag_input:
        ptag = ptag_input
    if ctag_input:
        ctag = ctag_input

    features = []
    words_tags_dict_100 = Feature2idClass.array_of_words_tags_dicts[0]
    words_tags_dict_101 = Feature2idClass.array_of_words_tags_dicts[1]
    words_tags_dict_102 = Feature2idClass.array_of_words_tags_dicts[2]
    threesome_tags_dict_103 = Feature2idClass.array_of_words_tags_dicts[3]
    couple_tags_dict_104 = Feature2idClass.array_of_words_tags_dicts[4]
    tags_dict_105 = Feature2idClass.array_of_words_tags_dicts[5]
    words_tags_dict_106 = Feature2idClass.array_of_words_tags_dicts[6]
    words_tags_dict_107 = Feature2idClass.array_of_words_tags_dicts[7]
    tag_and_previous_two_words_dict_f3 = Feature2idClass.array_of_words_tags_dicts[8]
    # words_tags_dict_capital_letter = Feature2idClass.array_of_words_tags_dicts[9]
    foursome_tags_dict = Feature2idClass.array_of_words_tags_dicts[9]
    tag_word_prev_word_dict = Feature2idClass.array_of_words_tags_dicts[10]
    first_tag_in_sentence_dict = Feature2idClass.array_of_words_tags_dicts[11]

    min_length_of_suf_pre_fix = Feature2idClass.min_length_of_suf_pre_fix
    max_length_suf_pre_fix = Feature2idClass.max_length_suf_pre_fix

    # 100 #
    if (cword, ctag) in words_tags_dict_100:
        features.append(words_tags_dict_100[(cword, ctag)])

    # 101 #
    for suffix_length in range(- max_length_suf_pre_fix, 1 - min_length_of_suf_pre_fix):
        i_letter_suffix = cword[suffix_length:]
        if (i_letter_suffix, ctag) in words_tags_dict_101:
            features.append(words_tags_dict_101[(i_letter_suffix, ctag)])

    # 102 #
    for prefix_length in range(min_length_of_suf_pre_fix, max_length_suf_pre_fix + 1):
        i_letter_prefix = cword[:prefix_length]
        if (i_letter_prefix, ctag) in words_tags_dict_102:
            features.append(words_tags_dict_102[(i_letter_prefix, ctag)])

    # 103 #
    three_consecutive_tags = (pptag, ptag, ctag)
    if three_consecutive_tags in threesome_tags_dict_103:
        features.append(threesome_tags_dict_103[three_consecutive_tags])

    # 104 #
    couple_consecutive_tags = (ptag, ctag)
    if couple_consecutive_tags in couple_tags_dict_104:
        features.append(couple_tags_dict_104[couple_consecutive_tags])

    # 105 #
    if ctag in tags_dict_105:
        features.append(tags_dict_105[ctag])

    # 106 #
        if (pword, ctag) in words_tags_dict_106:
            features.append(words_tags_dict_106[(pword, ctag)])

    # 107 #
        if (nword, ctag) in words_tags_dict_107:
            features.append(words_tags_dict_107[(nword, ctag)])

    # f3 #
    tag_and_previous_two_words = (ppword, pword, ctag)
    if tag_and_previous_two_words in tag_and_previous_two_words_dict_f3:
        features.append(tag_and_previous_two_words_dict_f3[tag_and_previous_two_words])

    # # capital letter #
    # if (cword, ctag) in words_tags_dict_capital_letter:
    #     features.append(words_tags_dict_capital_letter[(cword, ctag)])

    # foursome tags #
    four_consecutive_tags = (pptag, ptag, ctag, ntag)
    if four_consecutive_tags in foursome_tags_dict:
        features.append(foursome_tags_dict[four_consecutive_tags])

    # tag curr word, prev word #
    tag_word_prev_word = (pword, cword, ctag)
    if tag_word_prev_word in tag_word_prev_word_dict:
        features.append(tag_word_prev_word_dict[tag_word_prev_word])

    # first tag in sentence #
    if ctag in first_tag_in_sentence_dict:
        features.append(first_tag_in_sentence_dict[ctag])

    return features




def represent_input_with_features_for_test(history, Feature2idClass, num_features,
                                           history_tags_features_table, ind,
                                           ctag_input=None, pptag_input=None, ptag_input=None):
    """
        Extract feature vector in per a given history
        :param history: touple{ppword, pptag, pword, ptag, cword, ctag, nword, ntag}
        :param Feature2idClass - in order to be able to reach easily all its methods
        :param ctag_input
        :param pptag_input
        :param ptag input
        pay attention to the order!!!
            Return a list with all features that are relevant to the given history in a numpy array format, for runtime
            enhancement
    """
    ppword = history[0]
    pptag = history[1]
    pword = history[2]
    ptag = history[3]
    cword = history[4]
    ctag = history[5]
    nword = history[6]
    ntag = history[7]

    if pptag_input:
        pptag = pptag_input
    if ptag_input:
        ptag = ptag_input
    if ctag_input:
        ctag = ctag_input

    min_length_of_suf_pre_fix = Feature2idClass.min_length_of_suf_pre_fix
    max_length_suf_pre_fix = Feature2idClass.max_length_suf_pre_fix
    features = history_tags_features_table[ind]
    curr_index = 0
    words_tags_dict_100 = Feature2idClass.array_of_words_tags_dicts[0]
    words_tags_dict_101 = Feature2idClass.array_of_words_tags_dicts[1]
    words_tags_dict_102 = Feature2idClass.array_of_words_tags_dicts[2]
    threesome_tags_dict_103 = Feature2idClass.array_of_words_tags_dicts[3]
    couple_tags_dict_104 = Feature2idClass.array_of_words_tags_dicts[4]
    tags_dict_105 = Feature2idClass.array_of_words_tags_dicts[5]
    words_tags_dict_106 = Feature2idClass.array_of_words_tags_dicts[6]
    words_tags_dict_107 = Feature2idClass.array_of_words_tags_dicts[7]
    tag_and_previous_two_words_dict_f3 = Feature2idClass.array_of_words_tags_dicts[8]
    # words_tags_dict_capital_letter = Feature2idClass.array_of_words_tags_dicts[9]
    foursome_tags_dict = Feature2idClass.array_of_words_tags_dicts[9]
    tag_word_prev_word_dict = Feature2idClass.array_of_words_tags_dicts[10]
    first_tag_in_sentence_dict = Feature2idClass.array_of_words_tags_dicts[11]

    # 100 #
    if (cword, ctag) in words_tags_dict_100:
        features[curr_index] = words_tags_dict_100[(cword, ctag)]

    # 101 #
    for suffix_length in range(- max_length_suf_pre_fix, 1 - min_length_of_suf_pre_fix):
        curr_index += 1
        i_letter_suffix = cword[suffix_length:]
        if (i_letter_suffix, ctag) in words_tags_dict_101:
            features[curr_index] = words_tags_dict_101[(i_letter_suffix, ctag)]

    # 102 #
    for prefix_length in range(min_length_of_suf_pre_fix, max_length_suf_pre_fix + 1):
        curr_index += 1
        i_letter_prefix = cword[:prefix_length]
        if (i_letter_prefix, ctag) in words_tags_dict_102:
            features[curr_index] = words_tags_dict_102[(i_letter_prefix, ctag)]

    # 103 #
    curr_index += 1
    three_consecutive_tags = (pptag, ptag, ctag)
    if three_consecutive_tags in threesome_tags_dict_103:
        features[curr_index] = threesome_tags_dict_103[three_consecutive_tags]

    # 104 #
    curr_index += 1
    couple_consecutive_tags = (ptag, ctag)
    if couple_consecutive_tags in couple_tags_dict_104:
        features[curr_index] = couple_tags_dict_104[couple_consecutive_tags]

    # 105 #
    curr_index += 1
    if ctag in tags_dict_105:
        features[curr_index] = tags_dict_105[ctag]

    # 106 #
    curr_index += 1
    if (pword, ctag) in words_tags_dict_106:
        features[curr_index] = words_tags_dict_106[(pword, ctag)]

    # 107 #
    curr_index += 1
    if (nword, ctag) in words_tags_dict_107:
        features[curr_index] = words_tags_dict_107[(nword, ctag)]

    # f3 #
    curr_index += 1
    tag_and_previous_two_words = (ppword, pword, ctag)
    if tag_and_previous_two_words in tag_and_previous_two_words_dict_f3:
        features[curr_index] = tag_and_previous_two_words_dict_f3[tag_and_previous_two_words]

    # # capital letter #
    # curr_index += 1
    # if (cword, ctag) in words_tags_dict_capital_letter:
    #     features[curr_index] = words_tags_dict_capital_letter[(cword, ctag)]

    # four consecutive tags #
    curr_index += 1
    four_consecutive_tags = (pptag, ptag, ctag, ntag)
    if four_consecutive_tags in foursome_tags_dict:
        features[curr_index] = foursome_tags_dict[four_consecutive_tags]

    # tag, word and prev word #
    curr_index += 1
    tag_word_prev_word = (pword, cword, ctag)
    if tag_word_prev_word in tag_word_prev_word_dict:
        features[curr_index] = tag_word_prev_word_dict[tag_word_prev_word]

    # first tag in sentence #
    curr_index += 1
    if ctag in first_tag_in_sentence_dict:
        features[curr_index] = first_tag_in_sentence_dict[ctag]


    return features


def collect_history_quadruples(file_path):
    """

    :param file_path:
    :return: table of all histories in length 4 from text (the same number of words in text)
    """
    history_table = []
    with open(file_path) as f:
        for line_num, line in enumerate(f):
            splited_words = line.split(' ')
            del splited_words[-1]
            num_words_in_line = len(splited_words)
            for word_idx in range(0, num_words_in_line):
                if word_idx == 0:
                    ppword, pptag = ('*', '*')
                    pword, ptag = ('*', '*')
                    if word_idx == num_words_in_line - 1:
                        nword, ntag = ('STOP', 'STOP')
                    else:
                        nword, ntag = splited_words[word_idx + 1].split('_')

                elif word_idx == 1:
                    ppword, pptag = ('*', '*')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    if word_idx == num_words_in_line - 1:
                        nword, ntag = ('STOP', 'STOP')
                    else:
                        nword, ntag = splited_words[word_idx + 1].split('_')

                else:
                    ppword, pptag = splited_words[word_idx - 2].split('_')
                    pword, ptag = splited_words[word_idx - 1].split('_')
                    if word_idx == num_words_in_line - 1:
                        nword, ntag = ('STOP', 'STOP')
                    else:
                        nword, ntag = splited_words[word_idx + 1].split('_')

                cword, ctag = splited_words[word_idx].split('_')


                curr_quadruple_history = (ppword, pptag, pword, ptag, cword, ctag, nword, ntag)
                history_table.append((line_num, curr_quadruple_history))
                # if line_num == 322:
                #     print(word_idx)
                #     print(splited_words)
                #     print(curr_quadruple_history)


    return history_table


def generate_table_of_history_tags_features_for_training(my_feature2id_class, history_quadruple_table, tags_list):
    """

    :param my_feature2id_class:
    :param history_quadruple_table:
    :param tags_list:
    :return: table of features for every ctag X history
    """
    num_history_quadruple_elements = len(history_quadruple_table)
    amount_of_tags = len(tags_list)
    history_tags_features_table_for_training = np.empty((num_history_quadruple_elements, amount_of_tags), dtype=object)

    matrix_size = num_history_quadruple_elements * amount_of_tags
    # element_counter = 0
    for history_index, curr_history_quadruple in enumerate(history_quadruple_table):
        for ctag_index, ctag in enumerate(tags_list):
            curr_feature_vector = represent_input_with_features(curr_history_quadruple[1], my_feature2id_class, ctag)
            history_tags_features_table_for_training[history_index, ctag_index] = curr_feature_vector
            # element_counter += 1
            # if element_counter % (round(matrix_size / 10)) == 0:
            #     print(f'{round(100 * element_counter / matrix_size)}% finished')

    return history_tags_features_table_for_training


def get_table_of_features_for_given_history_num(my_feature2id_class, history_quadruple_table, tags_list, history_num,
                                                num_features):
    """

    :param my_feature2id_class:
    :param history_quadruple_table:
    :param tags_list:
    :param history_num:
    :return:
    1) table of features for every possibility of pptag X ptag X ctag X history that corresponds to history num
    2) the number of sentence in which this history appears
    3) boolean - is this the last word in the sentence
    """
    # num_history_quadruple_elements = len(history_quadruple_table)
    amount_of_tags = len(tags_list)
    asterisk = '*'
    asterisk_index = tags_list.index(asterisk)
    # progress_counter = 0

    curr_history_quadruple = history_quadruple_table[history_num]
    # print(curr_history_quadruple[1])
    # print(curr_history_quadruple)
    # if curr_history_quadruple[0] == 322:
    #     print(curr_history_quadruple)
    # if curr word is the beginning of the sentence, allow previous two tags to be asterisk only #
    if curr_history_quadruple[1][0] == asterisk and curr_history_quadruple[1][2] == asterisk:
        history_tags_features_table = np.full((1, 1, amount_of_tags, num_features), -1, dtype=np.int32)
        table_total_num_different_entries = amount_of_tags
        pptag = asterisk
        pptag_index = asterisk_index
        ptag = asterisk
        ptag_index = asterisk_index
        for ctag_index, ctag in enumerate(tags_list):
            # curr_feature_vector = represent_input_with_features(curr_history_quadruple[1], my_feature2id_class,ctag, pptag, ptag)
            curr_feature_vector = represent_input_with_features_for_test(curr_history_quadruple[1],
                                                                                 my_feature2id_class, num_features,
                                                                                 history_tags_features_table,
                                                                                 (pptag_index, ptag_index, ctag_index),
                                                                                 ctag, pptag, ptag)
            # progress_counter += 1
            # if progress_counter % (round(table_total_num_different_entries / 10)) == 0:
            #     print(f'{round(100 * progress_counter / table_total_num_different_entries)}% finished')

    # if curr word is the second word of the sentence, allow the tag of the word which is 2 words behind to be asterisk only #
    elif curr_history_quadruple[1][0] == asterisk:
        history_tags_features_table = np.full((1, amount_of_tags, amount_of_tags, num_features), -1, dtype=np.int32)
        table_total_num_different_entries = amount_of_tags * amount_of_tags
        pptag = asterisk
        pptag_index = asterisk_index
        for ptag_index, ptag in enumerate(tags_list):
            for ctag_index, ctag in enumerate(tags_list):
                # curr_feature_vector = represent_input_with_features(curr_history_quadruple[1], my_feature2id_class,ctag, pptag, ptag)
                curr_feature_vector = represent_input_with_features_for_test(curr_history_quadruple[1],
                                                                                 my_feature2id_class, num_features,
                                                                                 history_tags_features_table,
                                                                                 (pptag_index, ptag_index, ctag_index),
                                                                                 ctag, pptag, ptag)
                # progress_counter += 1
                # if progress_counter % (round(table_total_num_different_entries / 10)) == 0:
                #     print(f'{round(100 * progress_counter / table_total_num_different_entries)}% finished')

    # for the third word in the sentence and after, no problems for previous tags, therefore insert to table all possibilities #
    else:
        history_tags_features_table = np.full((amount_of_tags, amount_of_tags, amount_of_tags, num_features), -1, dtype=np.int32)
        table_total_num_different_entries = amount_of_tags * amount_of_tags * amount_of_tags
        for pptag_index, pptag in enumerate(tags_list):
            for ptag_index, ptag in enumerate(tags_list):
                for ctag_index, ctag in enumerate(tags_list):
                    # curr_feature_vector = represent_input_with_features(curr_history_quadruple[1], my_feature2id_class,ctag, pptag, ptag)
                    curr_feature_vector = represent_input_with_features_for_test(curr_history_quadruple[1],
                                                                                 my_feature2id_class, num_features,
                                                                                 history_tags_features_table,
                                                                                 (pptag_index, ptag_index, ctag_index),
                                                                                 ctag, pptag, ptag)
                    # progress_counter += 1
                    # if progress_counter % (round(table_total_num_different_entries / 10)) == 0:
                    #     print(f'{round(100 * progress_counter / table_total_num_different_entries)}% finished')

    is_last_word_in_sentence = False
    if curr_history_quadruple[1][6] == 'STOP':
        is_last_word_in_sentence = True
    return history_tags_features_table, curr_history_quadruple[0], is_last_word_in_sentence


def get_beam_of_features_for_given_history_num(my_feature2id_class, history_quadruple_table,
                                               tags_list, history_num, num_features, requests, beam_width):
    """

    :param my_feature2id_class:
    :param history_quadruple_table:
    :param tags_list:
    :param history_num:
    :return:
    1) table of features for every possibility of pptag X ptag X ctag X history that corresponds to history num
    2) the number of sentence in which this history appears
    3) boolean - is this the last word in the sentence
    """
    # num_history_quadruple_elements = len(history_quadruple_table)
    amount_of_tags = len(tags_list)
    asterisk = '*'
    curr_history_quadruple = history_quadruple_table[history_num]
    if curr_history_quadruple[1][0] == asterisk and curr_history_quadruple[1][2] == asterisk:
        num_pp, num_p = 1, 1
    elif curr_history_quadruple[1][0] == asterisk:
        num_pp, num_p = 1, amount_of_tags
    else:
        num_pp, num_p = amount_of_tags, amount_of_tags
    history_tags_features_table = np.full((beam_width, amount_of_tags, num_features), -1, dtype=np.int32)
    for i, (pptag_index, ptag_index) in enumerate(requests):
        for ctag_index, ctag in enumerate(tags_list):
            pptag, ptag = tags_list[pptag_index], tags_list[ptag_index]
            represent_input_with_features_for_test(curr_history_quadruple[1],
                                                   my_feature2id_class, num_features,
                                                   history_tags_features_table,
                                                   (i, ctag_index),
                                                   ctag, pptag, ptag)
    is_last_word_in_sentence = (curr_history_quadruple[1][6] == 'STOP')
    return history_tags_features_table, curr_history_quadruple[0], is_last_word_in_sentence, num_pp, num_p


def get_all_gt_tags_ordered(file_path):
    """

    :param file_path:
    :return: all tags in the text, in order of appearance
    """
    all_tags_gt_ordered = []
    with open(file_path) as f:
        for line in f:
            splited_words = line.split(' ')
            del splited_words[-1]
            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                all_tags_gt_ordered.append(cur_tag)
    return all_tags_gt_ordered


def get_all_words_ordered(file_path):
    """

    :param file_path:
    :return: all tags in the text, in order of appearance
    """
    all_words_ordered = []
    with open(file_path) as f:
        for line in f:
            splited_words = line.split(' ')
            del splited_words[-1]
            for word_idx in range(len(splited_words)):
                cur_word, cur_tag = splited_words[word_idx].split('_')
                all_words_ordered.append(cur_word)
    return all_words_ordered


def find_differences_in_possible_tags(file_path1, file_path2):
    tags1 = set(get_all_gt_tags_ordered(file_path1))
    tags2 = set(get_all_gt_tags_ordered(file_path2))
    diff = tags2 - tags1
    return tags1, tags2, diff


def sentence_index_from_history_table(history_table):
    curr_sentence = history_table[0][0]
    res = [[0]]
    for i, ent in enumerate(history_table):
        if ent[0] != curr_sentence:
            res[curr_sentence].append(i)
            res.append([i])
            curr_sentence += 1
    res[curr_sentence].append(len(history_table))
    return res


def k_fold_cross_validation(history_table, history_tag_table, sentence_indexs,
                            training_procedure, tester, true_tags, k=5):
    res = np.zeros(k)
    num_s = len(sentence_indexs)
    fold_size = int(num_s/k)
    for i in range(k):
        fold_start = sentence_indexs[i * fold_size][0]
        fold_end = sentence_indexs[(i + 1) * fold_size - 1][1]
        test = history_table[fold_start: fold_end]
        train = np.append(history_tag_table[: fold_start], history_tag_table[fold_end:], axis=0)
        true_tags_train = np.append(true_tags[: fold_start], true_tags[fold_end:])
        v = training_procedure(train, true_tags_train)
        res[i] = tester(test, true_tags[fold_start: fold_end], v)
    return res


def train_models(weights_path, feature_path, model):
    min_length_of_suf_pre_fix = 1
    max_length_suf_pre_fix = 4
    num_dicts = 11  # number of different feauture types
    num_additional_features = (max_length_suf_pre_fix - min_length_of_suf_pre_fix + 1) * 2
    num_features = num_dicts + num_additional_features
    if model == 'small':
        alpha, thresholds, beam = 0.1, 2, 2
        features_list = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]
        file_path = os.path.join("data", "train2.wtag")
    if model == 'big':
        alpha, thresholds, beam = 0.35, 0, 2
        features_list = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        file_path = os.path.join("data", "train1.wtag")
    num_occurrences_thresholds = np.ones(num_features) * thresholds

    my_feature_statistics_class = FeatureStatisticsClass(num_dicts)

    my_feature_statistics_class.get_word_tag_pair_count_100(file_path)
    my_feature_statistics_class.get_word_tag_pair_count_101(file_path, min_length_of_suf_pre_fix,
                                                            max_length_suf_pre_fix)
    my_feature_statistics_class.get_word_tag_pair_count_102(file_path, min_length_of_suf_pre_fix,
                                                            max_length_suf_pre_fix)
    my_feature_statistics_class.get_tag_threesome_count_103(file_path)
    my_feature_statistics_class.get_tag_couples_count_104(file_path)
    my_feature_statistics_class.get_tag_count_105(file_path)
    my_feature_statistics_class.get_prev_word_curr_tag_pair_count_106(file_path)
    my_feature_statistics_class.get_next_word_curr_tag_pair_count_107(file_path)
    my_feature_statistics_class.get_tag_threesome_count_f3(file_path)
    my_feature_statistics_class.get_tag_threesome_count_tag_cur_word_prev_word(file_path)
    my_feature2id_class = Feature2idClass(my_feature_statistics_class, num_occurrences_thresholds, num_features,
                                          min_length_of_suf_pre_fix, max_length_suf_pre_fix)
    my_feature2id_class.get_id_for_features_over_threshold(file_path, num_features, min_length_of_suf_pre_fix,
                                                           max_length_suf_pre_fix, features_list)
    train_tags_ordered = get_all_gt_tags_ordered(file_path)
    tags_list = []
    tags_list = list(tags_list)
    tags_list.append('*')
    tag_set = set(train_tags_ordered)
    tag_to_ind = {'*': 0}
    for i, tag in enumerate(tag_set):
        tags_list.append(tag)
        tag_to_ind[tag] = i + 1
    my_feature2id_class.tag_list = tags_list
    my_feature2id_class.tag_to_ind = tag_to_ind
    my_feature2id_class.num_feature_class = num_features
    history_quadruple_table = collect_history_quadruples(file_path)
    true_tags_train = np.array([tag_to_ind[x] for x in train_tags_ordered])
    history_tags_features_table_for_training = generate_table_of_history_tags_features_for_training(my_feature2id_class,
                                                                                                    history_quadruple_table,
                                                                                                    tags_list)

    if feature_path is not None:
        with open(feature_path, 'wb') as f:
            pickle.dump(my_feature2id_class, f)
    v = train_from_list(history_tags_features_table_for_training, true_tags_train, alpha, time_run=True,
                        weights_path=weights_path)
    return v, my_feature2id_class


def use_trained_model(weights_path, feature_path):
    with open(weights_path, 'rb') as f:
        v = pickle.load(f)[0]
    with open(feature_path, 'rb') as f:
        my_feature2id_class = pickle.load(f)
    tags_list = my_feature2id_class.tag_list
    tag_to_ind = my_feature2id_class.tag_to_ind
    num_features = my_feature2id_class.num_feature_class
    '''rest of the code here for example testing on test1'''
    test_path = os.path.join("data", "test1.wtag")
    test_history_quadruple_table = collect_history_quadruples(test_path)
    test_tags_ordered = get_all_gt_tags_ordered(test_path)
    true_tags_test = np.array([tag_to_ind.get(x, -1) for x in test_tags_ordered])
    sentence_indexes = sentence_index_from_history_table(test_history_quadruple_table)
    mat_gen = lambda h, requests, beam_width: get_beam_of_features_for_given_history_num(my_feature2id_class,
                                                                                         test_history_quadruple_table,
                                                                                         tags_list, h, num_features,
                                                                                         requests, beam_width)
    get_test_statistics(true_tags_test, mat_gen, v, 2, sentence_indexes)

def main():
    start_time_section_1 = time.time()
    # best pararms small model: alpha = 0.1, thresholds = 2, beam = 2, score = 0.934302, features = all first 8
    # best params big model: alpha = 0.35, thresholds = 0 beam =2, score = 0.9557, features = all first 8 without 103
    # defining hyper-parameters #
    min_length_of_suf_pre_fix = 1
    max_length_suf_pre_fix = 4
    alpha_big_model = 0.1  # regularization term (0.1 best)
    alpha_small_model = 0.05  # (0.000001 best)
    my_alpha = alpha_small_model
    num_dicts = 11  # number of different feauture types
    num_additional_features = (max_length_suf_pre_fix - min_length_of_suf_pre_fix + 1) * 2
    num_features = num_dicts + num_additional_features  # total number of different features (different lengths for suff/prefixes)
    features_list = [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0]   # binary array that determines which features are participating
    my_num_occurrences_thresholds = np.ones(num_features)  # vector of thresholds
    #num_occurrences_thresholds[5 + num_additional_features] = 2

    # features_list = np.random.randint(2, size=num_dicts)
    scores = {}
    file_path = os.path.join("data", "train2.wtag")
    test_path = os.path.join("data", "train2.wtag")

    # tags1, tags2, diff = find_differences_in_possible_tags(file_path, test_path)
    # all_words_in_text = get_all_words_ordered(file_path)
    # all_words_unique = set(all_words_in_text)

    # generate statistic class and count all features #
    my_feature_statistics_class = FeatureStatisticsClass(num_dicts)

    my_feature_statistics_class.get_word_tag_pair_count_100(file_path)
    my_feature_statistics_class.get_word_tag_pair_count_101(file_path, min_length_of_suf_pre_fix, max_length_suf_pre_fix)
    my_feature_statistics_class.get_word_tag_pair_count_102(file_path, min_length_of_suf_pre_fix, max_length_suf_pre_fix)
    my_feature_statistics_class.get_tag_threesome_count_103(file_path)
    my_feature_statistics_class.get_tag_couples_count_104(file_path)
    my_feature_statistics_class.get_tag_count_105(file_path)
    my_feature_statistics_class.get_prev_word_curr_tag_pair_count_106(file_path)
    my_feature_statistics_class.get_next_word_curr_tag_pair_count_107(file_path)
    my_feature_statistics_class.get_tag_threesome_count_f3(file_path)
    # my_feature_statistics_class.get_tag_word_count_capital_letter(file_path)
    # my_feature_statistics_class.get_tag_foursome_count(file_path)
    my_feature_statistics_class.get_tag_threesome_count_tag_cur_word_prev_word(file_path)
    # my_feature_statistics_class.get_tag_is_first_in_sentence(file_path)


    curr_percentile = 5
    # percentiles = my_feature_statistics_class.get_percentiles(curr_percentile)


    # num_occurrences_thresholds = [x + 1 for x in percentiles]

    # generate indices for all features that appear above a specified threshold #

    for alpha_coeff in range(2, 3):
        for th_coeff in range(0, 1):
            for beam_width in range(2, 3):

                # alpha = my_alpha * math.pow(10, alpha_coeff)
                alpha = my_alpha * alpha_coeff
                num_occurrences_thresholds = my_num_occurrences_thresholds * th_coeff


                my_feature2id_class = Feature2idClass(my_feature_statistics_class, num_occurrences_thresholds, num_features, min_length_of_suf_pre_fix, max_length_suf_pre_fix)
                my_feature2id_class.get_id_for_features_over_threshold(file_path, num_features, min_length_of_suf_pre_fix, max_length_suf_pre_fix, features_list)

                # generate a history quadruple table for train #
                history_quadruple_table = collect_history_quadruples(file_path)

                # generate a history quadruple table for test #
                test_history_quadruple_table = collect_history_quadruples(test_path)

                # generate a list of all possible tags and make it indexed according to appearance order in the set of tags #
                train_tags_ordered = get_all_gt_tags_ordered(file_path)

                tags_list = []
                tags_list = list(tags_list)
                tags_list.append('*')
                tag_set = set(train_tags_ordered)
                num_tags = len(tag_set)
                tag_to_ind = {'*': 0}
                for i, tag in enumerate(tag_set):
                    tags_list.append(tag)
                    tag_to_ind[tag] = i + 1

                train_correct_tags_ordered_indexed = [tag_to_ind[x] for x in train_tags_ordered]

                # test tags
                test_tags_ordered = get_all_gt_tags_ordered(test_path)
                test_correct_tags_ordered_indexed = [tag_to_ind.get(x, -1) for x in test_tags_ordered]  # all indices of tags (in the unique tag set) in order of appearance in text

                # generate a table with entries: (history_quadruple, ctag), that contains a matching feature #
                history_tags_features_table_for_training = generate_table_of_history_tags_features_for_training(my_feature2id_class,
                                                                                                                history_quadruple_table,
                                                                                                                tags_list)
                end_time_section_1 = time.time()
                total_time = end_time_section_1 - start_time_section_1
                print(f'total time for section 1 is: {total_time} seconds')
                true_tags_train = np.array(train_correct_tags_ordered_indexed)
                true_tags_test = np.array(test_correct_tags_ordered_indexed)

                #beam_width = 7
                mat_gen = lambda h, requests, beam_width: get_beam_of_features_for_given_history_num(my_feature2id_class,
                                                                                                    test_history_quadruple_table,
                                                                                                    tags_list, h, num_features,
                                                                                                    requests, beam_width)

                # v = train_from_list(history_tags_features_table_for_training, true_tags_train, alpha, time_run=True)
                #
                # score = compute_accuracy_beam(true_tags_test, mat_gen, v, beam_width, time_run=True, iprint=500)
                # print(score)
                # # scores[(str(features_list))] = score
                # scores[((alpha_coeff, th_coeff, beam_width))] = score
                # # print(f'best features vector is {max(scores, key=scores.get)}')
                # # print(f' best  is: {scores[max(scores, key=scores.get)]}')
                #
                # # print("")

                training_procedure = lambda table, tags: train_from_list(table, tags, alpha,
                                                                         num_f=my_feature2id_class.n_total_features)

                def tester(table, true_tags, v):
                    mat_gen = lambda h, requests, beam_width: get_beam_of_features_for_given_history_num(my_feature2id_class,
                                                                                                         table,
                                                                                                         tags_list, h, num_features,
                                                                                                         requests, beam_width)
                    return compute_accuracy_beam(true_tags, mat_gen, v, beam_width)

                sentence_indexs = sentence_index_from_history_table(history_quadruple_table)
                score = (np.mean(k_fold_cross_validation(history_quadruple_table, history_tags_features_table_for_training,
                                                      sentence_indexs, training_procedure, tester,
                                                      true_tags_train)))
                print(score)
                scores[((alpha_coeff, th_coeff, beam_width))] = score


    best_params = max(scores, key=scores.get)
    print(f'params {best_params}')
    print(f' best params: alpha: {my_alpha * best_params[0] }, thresholds: {best_params[1] * my_num_occurrences_thresholds}, beam width: {best_params[2]}')
    print(f' best  is: {scores[max(scores, key=scores.get)]}')

if __name__ == '__main__':
    train_models('samll_weights', 'small_features', 'small')
