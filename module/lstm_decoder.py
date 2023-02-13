import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.mlp import MLP
from module.dropout import SharedDropout


class LSTMDecoder(nn.Module):
    def __init__(self, x_dims, h_dims, ccg_dims, n_tag, vocab, mlp_drop, rnn_drop):
        super(LSTMDecoder, self).__init__()
        input_dim = x_dims + ccg_dims
        PAD = vocab.get_padding_index('tag')

        self.lstm = nn.LSTM(input_dim, h_dims)
        self.ccg_lookup = torch.nn.Embedding(n_tag, ccg_dims, padding_idx=PAD)
        self.W = MLP(h_dims, n_tag, mlp_drop)
        self.bilstm_drop = SharedDropout(rnn_drop)
        self.hidden_size = h_dims
        self.vocab = vocab

    def forward(self, x, atom_truth, atom_mask, tag_truth):
        batch_size = x.shape[0]
        max_len_tag = atom_truth.shape[1]
        if self.training:
            losses = None
        else:
            output = ['' for _ in range(batch_size)]
            flag = [0 for _ in range(batch_size)]
            max_len_tag = 125
            predict = None
        atom_cnt = 0
        start_flag = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        while atom_cnt < max_len_tag:
            if not start_flag:
                predict = torch.tensor([0] * batch_size, device=device)
                ccg_vec = self.ccg_lookup(predict)
                h0 = torch.randn(1, batch_size, self.hidden_size, device=device)
                c0 = torch.randn(1, batch_size, self.hidden_size, device=device)
                hidden = (h0, c0)
            else:
                if self.training:
                    ccg_vec = self.ccg_lookup(atom_truth[:, atom_cnt - 1])
                else:
                    ccg_vec = self.ccg_lookup(predict)
            input = torch.cat((x, ccg_vec), 1).unsqueeze(0)
            out, hidden = self.lstm(input, hidden)
            out = self.W(out).squeeze(0)
            if self.training:
                y = out[atom_mask[:, atom_cnt]]
                target = atom_truth[:, atom_cnt][atom_mask[:, atom_cnt]]
                loss = F.cross_entropy(y, target, reduction='sum')
                losses = losses + loss if losses is not None else loss
            else:
                predict = torch.argmax(out, 1)
                pred = predict.tolist()
                for i in range(batch_size):
                    temp = self.vocab.get_token_from_index(pred[i], 'atom_ccg')
                    if flag[i] == 0 and temp == 'EOS':
                        flag[i] = 1
                    elif flag[i] == 0 and temp != 'EOS':
                        output[i] += temp
            atom_cnt += 1
            start_flag = 1
        if self.training:
            return losses/torch.sum(atom_mask)
        else:
            good, total = 0, batch_size
            tag_truth = tag_truth.tolist()
            for i in range(batch_size):
                if self.vocab.get_token_index(output[i], 'tag') == tag_truth[i]:
                    good += 1
            return good, total






