"""
Script defining the different modules of model's architecture

@author: Abinash Sinha
"""

import numpy as np

import copy
import math
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class PointWiseFeedForward(nn.Module):
    def __init__(self, args):
        super(PointWiseFeedForward, self).__init__()
        self.conv1d_1 = nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=(1,))
        self.activation = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(args.hidden_size, args.hidden_size, kernel_size=(1,))
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.conv1d_1(input_tensor.transpose(1, 2))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv1d_2(hidden_states.transpose(1, 2))
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = PointWiseFeedForward(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class SASEncoder(nn.Module):
    def __init__(self, args):
        super(SASEncoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            return hidden_states
        return all_encoder_layers


class DSSEncoder(nn.Module):
    def __init__(self, args):
        super(DSSEncoder, self).__init__()
        # self.sas_encoder = SASEncoder(args)
        # prototypical intention vector for each intention
        self.prototypes = nn.ParameterList([nn.Parameter(torch.randn(args.hidden_size) *
                                                         (1 / np.sqrt(args.hidden_size)))
                                            for _ in range(args.num_intents)])

        self.layernorm1 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm2 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm3 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm4 = LayerNorm(args.hidden_size, eps=1e-12)
        self.layernorm5 = LayerNorm(args.hidden_size, eps=1e-12)

        self.w = nn.Linear(args.hidden_size, args.hidden_size)

        self.b_prime = nn.Parameter(torch.zeros(args.hidden_size))
        # self.b_prime = BiasLayer(args.hidden_size, 'zeros')

        # individual alpha for each position
        self.alphas = nn.Parameter(torch.zeros(args.max_seq_length, args.hidden_size))

        self.beta_input_seq = nn.Parameter(torch.randn(args.num_intents, args.hidden_size) *
                                           (1 / np.sqrt(args.hidden_size)))

        self.beta_label_seq = nn.Parameter(torch.randn(args.num_intents, args.hidden_size) *
                                           (1 / np.sqrt(args.hidden_size)))

    def _intention_clustering(self,
                              z: torch.Tensor) -> torch.Tensor:
        """
        Method to measure how likely the primary intention at position i
        is related with kth latent category
        :param z:
        :return:
        """
        z = self.layernorm1(z)
        hidden_size = z.shape[2]
        exp_normalized_numerators = list()
        i = 0
        for prototype_k in self.prototypes:
            prototype_k = self.layernorm2(prototype_k)  # [D]
            numerator = torch.matmul(z, prototype_k)  # [B, S]
            exp_normalized_numerator = torch.exp(numerator / np.sqrt(hidden_size))  # [B, S]
            exp_normalized_numerators.append(exp_normalized_numerator)
            if i == 0:
                denominator = exp_normalized_numerator
            else:
                denominator = torch.add(denominator, exp_normalized_numerator)
            i = i + 1

        all_attentions_p_k_i = [torch.div(k, denominator)
                                for k in exp_normalized_numerators]  # [B, S] K times
        all_attentions_p_k_i = torch.stack(all_attentions_p_k_i, -1)  # [B, S, K]

        return all_attentions_p_k_i

    def _intention_weighting(self,
                             z: torch.Tensor) -> torch.Tensor:
        """
        Method to measure how likely primary intention at position i
        is important for predicting user's future intentions
        :param z:
        :return:
        """
        hidden_size = z.shape[2]
        keys_tilde_i = self.layernorm3(z + self.alphas)  # [B, S, D]
        keys_i = keys_tilde_i + torch.relu(self.w(keys_tilde_i))  # [B, S, D]
        query = self.layernorm4(self.b_prime + self.alphas[-1, :] + z[:, -1, :])  # [B, D]
        query = torch.unsqueeze(query, -1)  # [B, D, 1]
        numerators = torch.matmul(keys_i, query)  # [B, S, 1]
        exp_normalized_numerators = torch.exp(numerators / np.sqrt(hidden_size))
        sum_exp_normalized_numerators = exp_normalized_numerators.sum(1).unsqueeze(-1)  # [B, 1] to [B, 1, 1]
        all_attentions_p_i = exp_normalized_numerators / sum_exp_normalized_numerators  # [B, S, 1]
        all_attentions_p_i = all_attentions_p_i.squeeze(-1)  # [B, S]

        return all_attentions_p_i

    def _intention_aggr(self,
                        z: torch.Tensor,
                        attention_weights_p_k_i: torch.Tensor,
                        attention_weights_p_i: torch.Tensor,
                        is_input_seq: bool) -> torch.Tensor:
        """
        Method to aggregate intentions collected at all positions according
        to both kinds of attention weights
        :param z:
        :param attention_weights_p_k_i:
        :param attention_weights_p_i:
        :param is_input_seq:
        :return:
        """
        attention_weights_p_i = attention_weights_p_i.unsqueeze(-1)  # [B, S, 1]
        attention_weights = torch.mul(attention_weights_p_k_i, attention_weights_p_i)  # [B, S, K]
        attention_weights_transpose = attention_weights.transpose(1, 2)  # [B, K, S]
        if is_input_seq:
            disentangled_encoding = self.beta_input_seq + torch.matmul(attention_weights_transpose, z)
        else:
            disentangled_encoding = self.beta_label_seq + torch.matmul(attention_weights_transpose, z)

        disentangled_encoding = self.layernorm5(disentangled_encoding)

        return disentangled_encoding  # [K, D]

    def forward(self,
                is_input_seq: bool,
                z: torch.Tensor):

        attention_weights_p_k_i = self._intention_clustering(z)  # [B, S, K]
        attention_weights_p_i = self._intention_weighting(z)  # [B, S]
        disentangled_encoding = self._intention_aggr(z,
                                                     attention_weights_p_k_i,
                                                     attention_weights_p_i,
                                                     is_input_seq)

        return disentangled_encoding
