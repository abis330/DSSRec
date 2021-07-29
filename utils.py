"""
Script having utility functions usable by multiples scripts
in the project

@author: Abinash Sinha
"""

import argparse
import numpy as np
import pandas as pd
import math
import random
import os
from scipy.sparse import csr_matrix

import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--plot_dir', default='plot', type=str)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--data_name', default='MOOCCube', type=str)
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument('--ckp', default=30, type=int, help="pretrain epochs 10, 20, 30...")
    parser.add_argument("--seed", default=42, type=int)

    # model args
    parser.add_argument("--model_name", default='pretrain', type=str)
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2,
                        help="number of layers")
    parser.add_argument('--num_attention_heads', default=1, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5,
                        help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5,
                        help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    # optimization args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    # finetune args
    get_finetune_args(parser)
    # pretrain args
    get_pretrain_args(parser)

    args = parser.parse_args()

    return args


def get_finetune_args(parser):
    # train args
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument("--mode", type=str, default='full',
                        help='indicates whether to use full/negative sampling')
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")


def get_pretrain_args(parser):
    # pre train args
    parser.add_argument("--pre_epochs", type=int, default=300, help="number of pre_train epochs")
    parser.add_argument("--pre_batch_size", type=int, default=256)
    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--prop_sliding_window", type=float, default=0.5, help="fraction of sequence length"
                                                                               "for sliding and creating"
                                                                               "new sequences")
    parser.add_argument("--s2i_weight", type=float, default=1.0, help="seq2item loss weight")
    parser.add_argument("--s2s_weight", type=float, default=1.0, help="seq2seq loss weight")
    parser.add_argument('--num_intents', default=4, type=int)
    parser.add_argument('--lambda_', default=0.5, type=float)
    parser.add_argument('--loss_type', default='DSS', type=str,
                        help='DSS -> disentangled self-supervision, '
                             'DSS-2 -> disentangled self-supervision 2')


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            # 有一个指标增加了就认为是还在涨
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model, epoch):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model, epoch)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model, epoch)
            self.counter = 0

    def save_checkpoint(self, score, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model of {epoch+1}...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def get_user_seqs_csv(data_file):
    data_df = pd.read_csv(data_file)
    user_seq = []
    video_item_set = set()
    for _, row in data_df.iterrows():
        video_items = row['video_ids'].split(',')
        video_items = [int(item) for item in video_items]
        user_seq.append(video_items)
        video_item_set = video_item_set | set(video_items)

    num_users = len(user_seq)
    max_item = len(video_item_set)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix


def get_user_seqs_long_csv(data_file):
    """

    :param data_file:
    :return:
    user_seq:
        list of item sequences
    max_item:
        item with largest item id (number of items basically)

    """
    data_df = pd.read_csv(data_file)
    user_seq = []
    long_sequence = []
    item_set = set()
    user_set = set()
    for idx, row in data_df.iterrows():
        items = row['video_ids'].split(',')
        items = [int(item) for item in items]
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
        user_set.add(row['id'])
    max_item = len(item_set)
    print('Number of unique item ids', max_item)
    assert len(user_set) == len(user_seq)
    return user_seq, max_item, long_sequence


def get_user_seqs_and_sample(data_file, sample_file):
    data_df = pd.read_csv(data_file)
    user_seq = []
    item_set = set()
    for _, row in data_df.iterrows():
        items = row['video_ids'].split(',')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)

    max_item = len(item_set)

    sample_df = pd.read_csv(sample_file)
    sample_seq = []
    for _, row in sample_df.iterrows():
        items = row['video_ids'].split(',')
        items = [int(item) for item in items]
        sample_seq.append(items)
        item_set = item_set | set(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq


def get_item2attribute(data_file):

    df = pd.read_csv(data_file)
    item2attribute = dict(zip(df.video_id, df.course_id))
    attribute_set = set()
    for item, attribute in item2attribute.items():
        attribute_set.add(attribute)
    attribute_size = len(attribute_set)
    print('Number of unique attribute ids', attribute_size)# 331
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
