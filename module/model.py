import sys
import argparse
import torch
import torch.nn as nn
from antu.io.vocabulary import Vocabulary
from module.mlp_decoder import MLPDecoder
from module.lstm_decoder import LSTMDecoder
from transformers import AutoModel


class Model(nn.Module):

    def __init__(
            self,
            vocabulary: Vocabulary,
            cfg: argparse.Namespace):
        super(Model, self).__init__()

        # Load pre-trained model
        self.bert_layer = AutoModel.from_pretrained(cfg.BERT)

        if cfg.DEC == 'MLP':
            self.model = MLPDecoder(cfg.BERT_HID, len(vocabulary.vocab['tag']), cfg.MLP_DROP)
        elif cfg.DEC == 'LSTM':
            self.model = LSTMDecoder(cfg.BERT_HID, cfg.LSTM_HID, cfg.CCG_DIM,
            len(vocabulary.vocab['atom_ccg']), vocabulary, cfg.MLP_DROP, cfg.LSTM_DROP)
        self.model_type = cfg.DEC


    def forward(self, x):
        encoded_layers = self.bert_layer(x['token'], attention_mask=x['token_mask'])

        batch_size = x['tag_mask'].shape[0]

        probe_word = encoded_layers[0]
        bert_token_reprs = [
            layer[starts.nonzero().squeeze(1)][1:-1]
            for layer, starts in zip(probe_word, x['split'])]
        # reshape
        bert_token = torch.cat(bert_token_reprs, 0)

        tag_truth = [tag[mask.nonzero().squeeze(1)]
                     for tag, mask in zip(x['tag'], x['tag_mask'])]
        tag_truth = torch.cat(tag_truth, 0)

        if self.model_type == 'MLP':
            return self.model(bert_token, tag_truth)
        elif self.model_type == 'LSTM':
            atom_truth = [atom[mask.nonzero().squeeze(1)]
                          for atom, mask in zip(x['atom_tag'], x['tag_mask'])]
            atom_mask = [atom[mask.nonzero().squeeze(1)]
                         for atom, mask in zip(x['atom_mask'], x['tag_mask'])]
            atom_truth = torch.cat(atom_truth, 0)
            atom_mask = torch.cat(atom_mask, 0)
            return self.model(bert_token, atom_truth, atom_mask, tag_truth)



