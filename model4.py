import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter


class MultiHeadAttentionImputation(nn.Module):
    def __init__(self, num_features=13, num_heads=2):
        super(MultiHeadAttentionImputation, self).__init__()

        head_dim = 2  # A smaller head dimension
        embed_dim = head_dim * num_heads  # Ensures it's divisible by num_heads

        # If the calculation above doesn't result in a factor of num_features, adjust it
        if embed_dim % num_features != 0:
            # Adjust head_dim to fit exactly into the nearest multiple of num_features
            head_dim = (num_features // num_heads) + (num_features % num_heads != 0)
            embed_dim = head_dim * num_heads

        # Linear layer to adjust input features to the embedding dimension
        self.adjust_dims = nn.Linear(num_features, embed_dim)

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Linear layer to project back to original dimension
        self.linear = nn.Linear(embed_dim, num_features)

    def forward(self, x):
        # Adjust input feature dimension to match embed_dim
        x = self.adjust_dims(x)
        attn_output, _ = self.attention(x, x, x)
        # Project attention output back to original feature size
        output = self.linear(attn_output)
        return output

class FusionModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FusionModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.gate = nn.Linear(output_dim, 1)

    def forward(self, rnn_output, feature_output):
        # Concatenate feature and RNN outputs
        x = torch.cat((rnn_output, feature_output), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        gate = torch.sigmoid(self.gate(x))
        return gate


class InputTemporalDecay(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        return torch.exp(-gamma)


class RNNContext(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, seq_lengths):
        T_max = input.shape[1]  # batch x time x dims

        h = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
        hn = torch.zeros(input.shape[0], self.hidden_size).to(input.device)

        for t in range(T_max):
            h = self.rnn_cell(input[:, t, :], h)
            padding_mask = ((t + 1) <= seq_lengths).float().unsqueeze(1).to(input.device)
            hn = padding_mask * h + (1-padding_mask) * hn

        return hn


class Hybrid_IIN(nn.Module):
    def __init__(self, num_vars, hidden_size=58, context_hidden=28, dropout_rate=0.4):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        self.context_rnn = RNNContext(2 * self.num_vars, context_hidden)

        self.initial_hidden = nn.Linear(context_hidden, hidden_size)
        self.initial_cell_state = nn.Tanh()

        self.rnn_cell_forward = nn.LSTMCell(2 * num_vars + context_hidden, hidden_size)
        self.rnn_cell_backward = nn.LSTMCell(2 * num_vars + context_hidden, hidden_size)

        self.decay_inputs = InputTemporalDecay(input_size=num_vars)

        self.recurrent_impute = nn.Linear(2 * hidden_size, num_vars)

        # Replace MLPFeatureImputation with attention or FM
        self.feature_impute = MultiHeadAttentionImputation(num_vars)  # or FactorizationMachine(num_vars)
        self.fusion_module = FusionModule(2 * num_vars, num_vars)  # Dynamic fusion mechanism

        #self.fuse_imputations = nn.Linear(2 * num_vars, num_vars)

    def forward(self, data):
        seq_lengths = data['lengths']
        values = data['values']
        masks = data['masks']
        deltas = data['deltas']

        # compute context vector, h0 and c0
        T_max = values.shape[1]
        padding_masks = torch.cat(tuple(((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                                        for t in range(T_max)), dim=1)
        padding_masks = padding_masks.unsqueeze(2).repeat(1, 1, values.shape[2])  # pts x time_stamps x vars

        data_means = values.sum(dim=1) / masks.sum(dim=1)  # pts x vars

        # normalization
        min_max_norm = data['max_vals'] - data['min_vals']
        normalized_values = (values - data['min_vals']) / min_max_norm
        normalized_means = (data_means - data['min_vals'].squeeze(1)) / min_max_norm.squeeze(1)


        if self.training:
            normalized_evals = (data['evals'] - data['min_vals']) / min_max_norm

        x_prime = torch.zeros_like(normalized_values)
        x_prime[:, 0, :] = normalized_values[:, 0, :]
        for t in range(1, T_max):
            x_prime[:, t, :] = normalized_values[:, t - 1, :]

        gamma = self.decay_inputs(deltas)
        x_decay = gamma * x_prime + (1 - gamma) * normalized_means.unsqueeze(1)
        x_complement = (masks * normalized_values + (1 - masks) * x_decay) * padding_masks
        decay_factors = self.decay_inputs(deltas)

        context_rnn = self.context_rnn(torch.cat((x_complement, deltas), dim=-1), seq_lengths)

        # Processing with forward and backward RNNs
        # Initialize hidden states for both forward and backward LSTMs
        h0_forward = self.initial_hidden(context_rnn)  # Initialize h0 for forward LSTM
        c0_forward = self.initial_cell_state(h0_forward)  # Initialize c0 for forward LSTM

        h0_backward = self.initial_hidden(context_rnn)  # Initialize h0 for backward LSTM
        c0_backward = self.initial_cell_state(h0_backward)  # Initialize c0 for backward LSTM

        forward_hidden_states = []
        backward_hidden_states = []
        h_forward, c_forward = h0_forward, c0_forward
        h_backward, c_backward = h0_backward, c0_backward
        inputs = torch.cat([x_complement, masks, context_rnn.unsqueeze(1).repeat(1, T_max, 1)], dim=-1)
        inputs = self.dropout(inputs)

        for t in range(max(seq_lengths)):
            h_forward, c_forward = self.rnn_cell_forward(inputs[:, t, :], (h_forward, c_forward))
            h_backward, c_backward = self.rnn_cell_backward(inputs[:, T_max - 1 - t, :], (h_backward, c_backward))

            forward_hidden_states.append(h_forward.unsqueeze(1))
            backward_hidden_states.append(h_backward.unsqueeze(1))

        forward_hidden_states = torch.cat(forward_hidden_states, dim=1)
        backward_hidden_states = torch.cat(backward_hidden_states[::-1], dim=1)

        # Combine hidden states and apply attention
        combined_hiddens = torch.cat((forward_hidden_states, backward_hidden_states), dim=2)
        attn_output = self.feature_impute(x_complement)
        combined_hiddens = self.dropout(combined_hiddens)

        # Imputation fusion
        rnn_imp = self.recurrent_impute(combined_hiddens)
        feat_imp = attn_output

        beta = self.fusion_module(rnn_imp, feat_imp)

        #beta = torch.sigmoid(self.fuse_imputations(torch.cat((decay_factors, masks), dim=-1)))
        imp_fusion = beta * feat_imp + (1 - beta) * rnn_imp
        final_imputation = masks * normalized_values + (1 - masks) * imp_fusion


        rnn_loss = F.l1_loss(rnn_imp * masks, normalized_values * masks, reduction='sum')
        feat_loss = F.l1_loss(feat_imp * masks, normalized_values * masks, reduction='sum')
        fusion_loss = F.l1_loss(imp_fusion * masks, normalized_values * masks, reduction='sum')
        total_loss = rnn_loss + feat_loss + fusion_loss

        if self.training:
            rnn_loss_eval = F.l1_loss(rnn_imp * data['eval_masks'], normalized_evals * data['eval_masks'],
                                       reduction='sum')
            feat_loss_eval = F.l1_loss(feat_imp * data['eval_masks'], normalized_evals * data['eval_masks'],
                                        reduction='sum')
            fusion_loss_eval = F.l1_loss(imp_fusion * data['eval_masks'], normalized_evals * data['eval_masks'],
                                          reduction='sum')
            total_loss_eval = rnn_loss_eval + feat_loss_eval + fusion_loss_eval

        def rescale(x):
            return torch.where(padding_masks == 1, x * min_max_norm + data['min_vals'], padding_masks)

        feat_imp = rescale(feat_imp)
        rnn_imp = rescale(rnn_imp)
        final_imp = rescale(final_imputation)

        out_dict = {
            'loss': total_loss / masks.sum(),
            'verbose_loss': [
                ('rnn_loss', rnn_loss / masks.sum(), masks.sum()),
                ('feat_loss', feat_loss / masks.sum(), masks.sum()),
                ('fusion_loss', fusion_loss / masks.sum(), masks.sum())
            ],
            'loss_count': masks.sum(),
            'imputations': final_imp,
            'feat_imp': feat_imp,
            'hist_imp': rnn_imp
        }
        if self.training:
            out_dict['loss_eval'] = total_loss_eval / data['eval_masks'].sum()
            out_dict['loss_eval_count'] = data['eval_masks'].sum()
            out_dict['verbose_loss'] += [
                ('rnn_loss_eval', rnn_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum()),
                ('feat_loss_eval', feat_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum()),
                ('fusion_loss_eval', fusion_loss_eval / data['eval_masks'].sum(), data['eval_masks'].sum())
            ]

        return out_dict