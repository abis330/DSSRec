"""
Script defining the model's architecture

@author: Abinash Sinha
"""

import numpy as np
import torch
import torch.nn as nn
from modules import DSSEncoder, SASEncoder, LayerNorm

torch.set_printoptions(profile="full")


class SASRecModel(nn.Module):
    """
    Class implementing SASRec model
    """

    def __init__(self, args):
        super().__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = SASEncoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.__init_weights)

    def _add_position_embedding(self,
                                sequence: torch.Tensor) -> torch.Tensor:

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def _get_embedding_and_mask(self, input_ids):
        sequence_emb = self._add_position_embedding(input_ids)
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return sequence_emb, extended_attention_mask

    def __init_weights(self, module):
        """
        Method to initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def finetune(self, input_ids):
        """
        Method to fine-tune the pre-trained model
        :param input_ids:
        :return:
        """
        sequence_emb, extended_attention_mask = self._get_embedding_and_mask(input_ids)

        sequence_output = self.item_encoder(sequence_emb,
                                            extended_attention_mask,
                                            output_all_encoded_layers=False)

        return sequence_output


class DSSRecModel(SASRecModel):
    """
    Version 1 of Disentangled Self-Supervision
    """

    def __init__(self, args):
        super().__init__(args)
        self.disentangled_encoder = DSSEncoder(args)

    # re-checked the math behind the code, it is CORRECT
    def __seq2seqloss(self,
                      inp_subseq_encodings: torch.Tensor,
                      label_subseq_encodings: torch.Tensor) -> torch.Tensor:
        sqrt_hidden_size = np.sqrt(self.args.hidden_size)
        product = torch.mul(inp_subseq_encodings, label_subseq_encodings)  # [B, K, D]
        normalized_dot_product = torch.sum(product, dim=-1) / sqrt_hidden_size  # [B, K]
        numerator = torch.exp(normalized_dot_product)  # [B, K]
        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        inp_subseq_encodings_trans_expanded = inp_subseq_encodings_trans.unsqueeze(1)  # [K, 1, B, D]
        label_subseq_encodings_trans = label_subseq_encodings.transpose(0, 1).transpose(1, 2)  # [K, D, B]
        dot_products = torch.matmul(inp_subseq_encodings_trans_expanded, label_subseq_encodings_trans)  # [K, K, B, B]
        dot_products = torch.exp(dot_products / sqrt_hidden_size)
        dot_products = dot_products.sum(-1)  # [K, K, B]
        temp = dot_products.sum(1)  # [K, B]
        denominator = temp.transpose(0, 1)  # [B, K]
        seq2seq_loss_k = -torch.log2(numerator / denominator)
        seq2seq_loss_k = torch.flatten(seq2seq_loss_k)
        thresh_th = int(np.floor(self.args.lambda_ * self.args.pre_batch_size * self.args.num_intents))
        thresh = torch.kthvalue(seq2seq_loss_k, thresh_th)[0]
        conf_indicator = seq2seq_loss_k <= thresh
        conf_seq2seq_loss_k = torch.mul(seq2seq_loss_k, conf_indicator)
        seq2seq_loss = torch.sum(conf_seq2seq_loss_k)
        # seq2seq_loss = torch.sum(seq2seq_loss_k)
        return seq2seq_loss

    # re-checked the math behind the code, it is CORRECT
    def __seq2itemloss(self,
                       inp_subseq_encodings: torch.Tensor,
                       next_item_emb: torch.Tensor) -> torch.Tensor:
        sqrt_hidden_size = np.sqrt(self.args.hidden_size)
        next_item_emb = torch.transpose(next_item_emb, 1, 2)  # [B, D, 1]
        dot_product = torch.matmul(inp_subseq_encodings, next_item_emb)  # [B, K, 1]
        exp_normalized_dot_product = torch.exp(dot_product / sqrt_hidden_size)
        numerator = torch.max(exp_normalized_dot_product, dim=1)[0]  # [B, 1]

        inp_subseq_encodings_trans = inp_subseq_encodings.transpose(0, 1)  # [K, B, D]
        next_item_emb_trans = next_item_emb.squeeze(-1).transpose(0, 1)  # [D, B]
        # sum of dot products of given input sequence encoding for each intent with all next item embeddings
        dot_products = torch.matmul(inp_subseq_encodings_trans,
                                    next_item_emb_trans) / sqrt_hidden_size  # [K, B, B]
        dot_products = torch.exp(dot_products)  # [K, B, B]
        dot_products = dot_products.sum(-1)
        dot_products = dot_products.transpose(0, 1)  # [B, K]
        # sum across all intents
        denominator = dot_products.sum(-1).unsqueeze(-1)  # [B, 1]
        seq2item_loss_k = -torch.log2(numerator / denominator)  # [B, 1]
        seq2item_loss = torch.sum(seq2item_loss_k)
        return seq2item_loss

    def pretrain(self,
                 inp_subseq: torch.Tensor,
                 label_subseq: torch.Tensor,
                 next_item: torch.Tensor):
        next_item_emb = self.item_embeddings(next_item)  # [B, 1, D]

        inp_subseq_emb, inp_subseq_ext_attn_mask = self._get_embedding_and_mask(inp_subseq)

        input_subseq_encoding = self.item_encoder(inp_subseq_emb,
                                                  inp_subseq_ext_attn_mask,
                                                  output_all_encoded_layers=False)

        label_subseq_emb, label_subseq_ext_attn_mask = self._get_embedding_and_mask(label_subseq)

        label_subseq_encoding = self.item_encoder(label_subseq_emb,
                                                  label_subseq_ext_attn_mask,
                                                  output_all_encoded_layers=False)
        # Encode masked sequence
        # inp_sequence_emb = self._add_position_embedding(inp_pos_items)
        # inp_sequence_mask = (inp_pos_items == 0).float() * -1e8
        # inp_sequence_mask = torch.unsqueeze(torch.unsqueeze(inp_sequence_mask, 1), 1)
        # label_sequence_emb = self._add_position_embedding(label_subseq)
        # label_sequence_mask = (label_subseq == 0).float() * -1e8
        # label_sequence_mask = torch.unsqueeze(torch.unsqueeze(label_sequence_mask, 1), 1)

        disent_inp_subseq_encodings = self.disentangled_encoder(True,
                                                                input_subseq_encoding)
        disent_label_seq_encodings = self.disentangled_encoder(False,
                                                               label_subseq_encoding)
        # seq2item loss
        seq2item_loss = self.__seq2itemloss(disent_inp_subseq_encodings, next_item_emb)
        # seq2seq loss
        seq2seq_loss = self.__seq2seqloss(disent_inp_subseq_encodings, disent_label_seq_encodings)

        return seq2item_loss, seq2seq_loss


class DSSRecModel2(DSSRecModel):
    """
    Version 2 of Disentangled Self-Supervision
    Here the K (number of intents) encodings collectively represent
    a sequence
    """

    def __init__(self, args):
        super().__init__(args)

    def finetune(self, input_ids):
        """
        Method to fine-tune the pre-trained model
        :param input_ids:
        :return:
        """
        sequence_emb, extended_attention_mask = self._get_embedding_and_mask(input_ids)

        item_encoded_layer = self.item_encoder(sequence_emb,
                                               extended_attention_mask,
                                               output_all_encoded_layers=False)
        sequence_encodings = self.disentangled_encoder(True,
                                                       item_encoded_layer)

        return sequence_encodings
