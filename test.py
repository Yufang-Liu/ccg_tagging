import argparse
import random

import torch
from torch.utils.data import DataLoader
import numpy as np
from module.model import Model
from utils.PTBDataset import PTBDataset, add_padding


def parse_args():
    """parse model configuration

    :return: cfg
    """
    parser = argparse.ArgumentParser(description="Experiment")
    # Data IO
    parser.add_argument('--TEST', type=str, help="Test set path.",
                        default='/home/yfliu/public/corpus/PTB_CCG/clean/wsj23.stagged')
    # model selection
    parser.add_argument('--DEC', type=str, help="Selection of decoder", default='LSTM')
    #Bert model cache
    parser.add_argument('--BERT', type=str, help="Bert model type", default='roberta-base')
    parser.add_argument('--CACHE', type=str, help="Cache file for Bert model",
                        default='./ckpts/bert_model')
    # Training setup
    parser.add_argument('--SEED', type=int, help="Set random seed.", default=666)
    parser.add_argument('--N_BATCH', type=int, help="Batch size for training & testing.", default=128)
    parser.add_argument('--N_WORKER', type=int, help="#Worker for data loader.", default=2)
    # Network setup
    parser.add_argument('--BERT_HID', type=int, help="Dimension of bert representation", default=768)
    parser.add_argument('--LSTM_HID', type=int, help="Dimension of lstm hidden representation", default=200)
    parser.add_argument('--CCG_DIM', type=int, help="Dimension of ccg word representation", default=30)
    parser.add_argument('--MLP_DROP', type=float, help="Dropout rate of MLP representation.", default=0.33)
    parser.add_argument('--EMB_DROP', type=float, help="Dropout rate of embedding representation.", default=0.33)
    parser.add_argument('--LSTM_DROP', type=float, help="Dropout rate of decoder lstm.", default=0.33)

    return parser.parse_args()


def main():
    # Configuration file processing
    cfg = parse_args()

    # Set seeds
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    BEST = './ckpts/' + cfg.DEC + '_best.pt'

    ckpt = torch.load(BEST)
    data_reader = ckpt['data_reader']
    vocabulary = ckpt['vocab']

    # Build data reader
    test_set = PTBDataset(cfg.TEST, data_reader, vocabulary, cfg)
    # Build the data-loader
    test = DataLoader(test_set, cfg.N_BATCH, False, num_workers=cfg.N_WORKER, pin_memory=cfg.N_WORKER > 0,
                      collate_fn=add_padding)

    # Build parser model
    model = Model(vocabulary, cfg)
    model.load_state_dict(ckpt['model'])

    # if running on GPU
    if torch.cuda.is_available(): model = model.cuda()

    @torch.no_grad()
    def validation(data_loader: DataLoader):
        good, total = 0, 0
        for data in data_loader:
            if cfg.N_WORKER:
                for x in data.keys(): data[x] = data[x].cuda()
            good_batch, total_batch = model(data)
            good += good_batch
            total += total_batch
        return good * 100.0 / total

    # validate prepare on dev set
    model.eval()
    acc = validation(test)
    print(acc)

if __name__ == '__main__':
    main()