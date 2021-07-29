"""
Script used for preparing dataset for pre-training and fine-tuning the model

@author: Abinash Sinha
"""

import torch
from torch.utils.data import Dataset

from utils import neg_sample


class PretrainDataset(Dataset):
    """
    Class for dataset preparation for self-supervised training
    """

    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.prop_sliding_window = args.prop_sliding_window
        self.loss_type = args.loss_type
        self.part_sequence = []
        self.loss_type = args.loss_type
        if self.loss_type in ['DSS', 'DSS-2']:
            self.__split_sequence_disent()
        else:
            self.__split_sequence()

    def __split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len + 2):-2]  # keeping same as train set
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[:(i + 1) + 1])

    def __split_sequence_disent(self):
        sliding_step = (int)(self.prop_sliding_window * self.max_len) \
            if self.prop_sliding_window != -1.0 else self.max_len
        for item_seq in self.user_seq:
            if len(item_seq) == 0:
                print("got empty seq:")
                continue
            if len(item_seq) <= self.max_len:
                self.part_sequence.append(item_seq[-(self.max_len + 2):-2])
            else:
                beg_idx = list(range(len(item_seq) - self.max_len, 0, -sliding_step))
                beg_idx.append(0)
                for i in beg_idx[::-1]:
                    self.part_sequence.append(item_seq[i - 2:i + self.max_len - 2])  # keeping same as train set

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self,
                    index: int) -> tuple:
        """

        :param index:
        :return:
        """
        # (all possible sequences made using all possible subsets of sequences)
        sequence = self.part_sequence[index]  # pos_items
        item_set = set(sequence)
        seq_len = len(sequence)
        # pad_len = self.max_len - len(sequence)
        if self.loss_type in ['DSS', 'DSS-2']:
            inp_subseq, label_subseq, next_item = self._get_input_label_subsequences(seq_len, sequence)
            cur_tensors = self.__get_items_dss_loss(inp_subseq, label_subseq, next_item)
        else:
            raise ValueError('No such loss is defined!')

        return cur_tensors

    def __get_items_dss_loss(self,
                             inp_subseq: list,
                             label_subseq: list,
                             next_item: list) -> tuple:
        """
        Method to prepare data instance for Disentangled Self-Supervision
        :param inp_subseq:
        :param label_subseq:
        :param next_item:
        :return:
        """
        # data preparation for DSS loss
        # input sub-sequence
        inp_pad_len = self.max_len - len(inp_subseq)
        inp_pos_items = ([0] * inp_pad_len) + inp_subseq
        inp_pos_items = inp_pos_items[-self.max_len:]
        # label sub-sequence
        len_label_subseq = len(label_subseq)
        label_subseq.reverse()
        label_pad_len = self.max_len - len_label_subseq
        label_pos_items = [0] * label_pad_len + label_subseq
        label_pos_items = label_pos_items[-self.max_len:]
        # label_pos_items.reverse()
        assert len(inp_pos_items) == self.max_len
        assert len(label_pos_items) == self.max_len
        # end of data preparation for DSS loss
        cur_tensors = (
            torch.tensor(inp_pos_items, dtype=torch.long),  # actual input sub-sequence of items
            torch.tensor(label_pos_items, dtype=torch.long),  # actual label sub-sequence of items
            torch.tensor(next_item, dtype=torch.long),  # item next to input sub-sequence of items
        )
        return cur_tensors

    @staticmethod
    def _get_input_label_subsequences(seq_len: int,
                                      sequence: list):
        if seq_len == 2:
            t = 1
        else:
            if not seq_len == 1:
                t = torch.randint(1, seq_len - 1, (1,))
        # input sub-sequence
        if seq_len == 1:
            inp_subseq = sequence
        else:
            inp_subseq = sequence[:t]
        # label sub-sequence
        if seq_len == 1:
            label_subseq = sequence
        else:
            label_subseq = sequence[t:]
        # next item
        if seq_len == 1:
            next_item = [sequence[0]]
        else:
            next_item = [sequence[t]]
        return inp_subseq, label_subseq, next_item


class FinetuneDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self,
                    index: int) -> tuple:

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return len(self.user_seq)
