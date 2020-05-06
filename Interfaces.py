import os
import pickle
import numpy as np
import part1_Defining_and_Extracting_Features as p1
import part2_optimization as p2


def train_model(file_path=None, alpha=0., threshold=2, features_list=None, our_model=None,
                weights_path=None, feature_path=None):
    """

    :param file_path: file to train from
    :param our_model: if 'big' train using out parameters for the big model
    :param alpha: the regularization parameter
    :param threshold: the threshold for feature appearances
    :param features_list: binary list [use f100,...,use f107, use f1, ..., usef4] if 0 not using the feature group
    :param weights_path: path to save the weights if None not saving
    :param feature_path: path to save the feature if None mot saving
    :return: the weight, class containing the feature dictionaries
    """
    min_length_of_suf_pre_fix = 1
    max_length_suf_pre_fix = 4
    num_dicts = 11  # number of different feauture types
    num_additional_features = (max_length_suf_pre_fix - min_length_of_suf_pre_fix + 1) * 2
    num_features = num_dicts + num_additional_features
    if our_model == 'small':
        alpha, threshold = 0.1, 2
        features_list = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0]
    if our_model == 'big':
        alpha, threshold = 0.35, 0
        features_list = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    num_occurrences_thresholds = np.ones(num_features) * threshold

    my_feature_statistics_class = p1.FeatureStatisticsClass(num_dicts)

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
    my_feature_statistics_class.get_tag_word_count_capital_letter(file_path)
    my_feature2id_class = p1.Feature2idClass(my_feature_statistics_class, num_occurrences_thresholds, num_features,
                                          min_length_of_suf_pre_fix, max_length_suf_pre_fix)
    my_feature2id_class.get_id_for_features_over_threshold(file_path, num_features, min_length_of_suf_pre_fix,
                                                           max_length_suf_pre_fix, features_list)
    train_tags_ordered = p1.get_all_gt_tags_ordered(file_path)
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
    history_quadruple_table = p1.collect_history_quadruples(file_path)
    true_tags_train = np.array([tag_to_ind[x] for x in train_tags_ordered])
    history_tags_features_table_for_training = p1.generate_table_of_history_tags_features_for_training(my_feature2id_class,
                                                                                                    history_quadruple_table,
                                                                                                    tags_list)

    if feature_path is not None:
        with open(feature_path, 'wb') as f:
            pickle.dump(my_feature2id_class, f)
    v = p2.train_from_list(history_tags_features_table_for_training, true_tags_train, alpha, time_run=True,
                           weights_path=weights_path)
    return v, my_feature2id_class


def test_model(test_path, read_model_from_file=True, weights_path=None, feature_path=None, weights=None,
               feature_class=None, beam=2, tagged_test_path=None):
    """
    :param test_path: path of the test file
    :param read_model_from_file: reading the model from file
    :param weights_path: path of pre trained weight used only if read_model_from_file is True
    :param feature_path: path of pre trained features used only if read_model_from_file is True
    :param weights: pre trained weight used only if read_model_from_file is False
    :param feature_class: pre trained features class used only if read_model_from_file is False
    :param beam: the beam width
    :param tagged_test_path: path to save the tagged test if None tagged test wont be saved
    :return: the accuracy
    """
    if read_model_from_file:
        with open(weights_path, 'rb') as f:
            v = pickle.load(f)[0]
        with open(feature_path, 'rb') as f:
            my_feature2id_class = pickle.load(f)
    else:
        v = weights
        my_feature2id_class = feature_class
    tag_infer = p1.infer_using_trained_model(None, None, '', test_path, read_from_file=False, v=v,
                                             my_feature2id_class=my_feature2id_class, beam=beam)
    keep_tagged_test = tagged_test_path is not None
    if not keep_tagged_test:
        tagged_test_path = 'temp'
    elif tagged_test_path[-4:] == '.wtag':
        tagged_test_path = tagged_test_path[: -4]
    p1.tag_file('', test_path, tagged_test_path, tag_infer)
    tagged_test_path = tagged_test_path + '.wtag'
    score = p1.compare_tagging_results(test_path, tagged_test_path)
    if not keep_tagged_test:
        os.remove(tagged_test_path)
    return score


def cross_validate_file(file_path, alpha=0., threshold=0, features_list=None, beam=2, print_stat=True, our_model=None):
    """
    cross validate the model
    :param our_model: if 'big' train using out parameters for the big model
    :param alpha: the regularization parameter
    :param threshold: the threshold for feature appearances
    :param features_list: binary list [use f100,...,use f107, use f1, ..., usef4] if 0 not using the feature group
    :param print_stat: if True print statistics for the accuracy on the folds
    :param file_path: path of the file
    :param beam: the beam width
    :param print_stat:
    :return: the mean accuracy
    """
    if our_model == 'small':
        alpha, threshold = 0.1, 2
        features_list = [1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0]
    if our_model == 'big':
        alpha, threshold = 0.35, 0
        features_list = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    score = p1.cross_validate(file_path=file_path, print_stat=print_stat, alpha=alpha, thresholds=threshold, beam=beam,
                              features_list=features_list)
    return score


def tag_file(file_path, read_model_from_file=True, weights_path=None, feature_path=None, weights=None,
             feature_class=None, beam=2, tagged_path=None):
    """
    :param file_path: path of the test file
    :param read_model_from_file: reading the model from file
    :param weights_path: path of pre trained weight used only if read_model_from_file is True
    :param feature_path: path of pre trained features used only if read_model_from_file is True
    :param weights: pre trained weight used only if read_model_from_file is False
    :param feature_class: pre trained features class used only if read_model_from_file is False
    :param beam: the beam width
    :param tagged_path: the path for the tagged file
    """
    test_model(file_path, read_model_from_file, weights_path, feature_path, weights, feature_class, beam, tagged_path)

