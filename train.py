import argparse
import os
import time
import random, math
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from antu.io.vocabulary import Vocabulary
from antu.utils.dual_channel_logger import dual_channel_logger
from module.model import Model
from utils.Reader import PTBReader
from utils.PTBDataset import PTBDataset, add_padding


def parse_args():
    """parse model configuration

    :return: cfg
    """
    parser = argparse.ArgumentParser(description="Experiment")
    # Data IO
    parser.add_argument('--TRAIN', type=str, help="Train set path.",
                        default='/home/yfliu/public/corpus/PTB_CCG/clean/wsj02-21.stagged')
    parser.add_argument('--DEV', type=str, help="Development set path.",
                        default='/home/yfliu/public/corpus/PTB_CCG/clean/wsj00.stagged')
    parser.add_argument('--TEST', type=str, help="Test set path.",
                        default='/home/yfliu/public/corpus/PTB_CCG/clean/wsj23.stagged')
    #Bert model cache
    parser.add_argument('--BERT', type=str, help="Bert model type", default='roberta-base')
    parser.add_argument('--CACHE', type=str, help="Cache file for Bert model",
                        default='./ckpts/bert_model')
    #model selection
    parser.add_argument('--DEC', type=str, help="Selection of decoder", default='MLP')
    # Training setup
    parser.add_argument('--SEED', type=int, help="Set random seed.", default=666)
    parser.add_argument('--N_EPOCH', type=int, help="#Epoch for training & testing.", default=10)
    parser.add_argument('--N_BATCH', type=int, help="Batch size for training & testing.", default=128)
    parser.add_argument('--N_WORKER', type=int, help="#Worker for data loader.", default=2)
    parser.add_argument('--IS_RESUME', default=False, action='store_true', help="Continue training.")
    # Optimizer
    parser.add_argument('--LR', type=float, help="Learning rate.", default=5e-5)
    parser.add_argument('--BETAS', type=float, nargs=2, help="Beta1 and Beta2 in Adam.", default=[0.9, 0.9])
    parser.add_argument('--EPS', type=float, help="EPS in Adam.", default=1e-8)
    #parser.add_argument('--LR_DECAY', type=float, help="Decay rate of LR.", default=0.9)
    #parser.add_argument('--LR_ANNEAL', type=int, help="Anneal step of LR.", default=5000)
    #parser.add_argument('--CLIP', type=float, help="Gradient clipping.", default=5.0)
    parser.add_argument('--LR_WARM_STEPS', type=int, help="Learning rate warm steps.", default=150)
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    LOG = './ckpts/' + cfg.DEC + '.log'
    LAST = './ckpts/' + cfg.DEC + '_last.pt'
    BEST = './ckpts/' + cfg.DEC + '_best.pt'

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=LOG,
        file_model='w',
        formatter='%(asctime)s - %(levelname)s - %(message)s',
        time_formatter='%m-%d %H:%M')

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Build data reader
    data_reader = PTBReader(field_list=['word', 'tag'])
    vocabulary = Vocabulary()
    counters = {'word': Counter(), 'tag': Counter(), 'atom_ccg': Counter()}
    # min_count = {'tag': 10}
    # Build the dataset
    train_set = PTBDataset(cfg.TRAIN, data_reader, vocabulary, cfg, counters) #, min_count)
    dev_set = PTBDataset(cfg.DEV, data_reader, vocabulary, cfg)
    test_set = PTBDataset(cfg.TEST, data_reader, vocabulary, cfg)
    # Build the data-loader
    train = DataLoader(train_set, cfg.N_BATCH, True, num_workers=cfg.N_WORKER, pin_memory=cfg.N_WORKER > 0,
                       collate_fn=add_padding)
    dev = DataLoader(dev_set, cfg.N_BATCH, False, num_workers=cfg.N_WORKER, pin_memory=cfg.N_WORKER > 0,
                     collate_fn=add_padding)
    test = DataLoader(test_set, cfg.N_BATCH, False, num_workers=cfg.N_WORKER, pin_memory=cfg.N_WORKER > 0,
                      collate_fn=add_padding)

    # Build model
    model = Model(vocabulary, cfg)

    # if running on GPU
    if torch.cuda.is_available(): model = model.cuda()

    # build optimizers
    if cfg.DEC == 'MLP':
        optim = AdamW(model.parameters(),
                     cfg.LR, cfg.BETAS, cfg.EPS)
    elif cfg.DEC == 'LSTM':
        optim = AdamW(
            [
                {'params': model.bert_layer.parameters()},
                {'params': model.model.parameters(), 'lr':0.01}
            ], cfg.LR, cfg.BETAS, cfg.EPS
        )
    training_step = math.ceil(len(train_set)/cfg.N_BATCH)
    sched = get_linear_schedule_with_warmup(optim, cfg.LR_WARM_STEPS, training_step*cfg.N_EPOCH)

    # load checkpoint if wanted
    start_epoch, best_acc, best_epoch = 0, 0, 0

    def load_ckpt(ckpt_path: str):
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']
        best_acc, best_epoch = ckpt['best']
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])

    if cfg.IS_RESUME: load_ckpt(LAST)

    @torch.no_grad()
    def validation(data_loader: DataLoader):
        good, total = 0, 0
        for data in data_loader:
            if cfg.N_WORKER:
                for x in data.keys(): data[x] = data[x].cuda()
            good_batch, total_batch = model(data)
            good += good_batch
            total += total_batch
        return good*100.0/total

    # Train model
    prepare_time_tot = process_time_tot = 0
    for epoch in range(start_epoch, cfg.N_EPOCH):
        model.train()
        tag_losses = []
        start_time = time.time()
        for data in train:
            if cfg.N_WORKER:
                for x in data.keys(): data[x] = data[x].cuda()
            prepare_time = time.time() - start_time
            prepare_time_tot += prepare_time
            optim.zero_grad()
            tag_loss = model(data)
            tag_losses.append(tag_loss.item())
            tag_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP)
            optim.step()
            process_time = time.time() - start_time - prepare_time
            process_time_tot += process_time
            start_time = time.time()
            sched.step()

        # save current model
        torch.save({
            'epoch': epoch,
            'best': (best_acc, best_epoch),
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'sched': sched.state_dict(),
            'data_reader': data_reader,
            'vocab': vocabulary,
        }, LAST)

        # validate parer on dev set
        model.eval()
        acc = validation(dev)
        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            os.popen(f'cp {LAST} {BEST}')
        logger.info(
            f'|{epoch:4}| Tag_loss({float(np.mean(tag_losses)):.2f}) '
            f'Best({best_epoch})')
        logger.info(f'| Dev| Acc:{acc:6.2f}')
        # view performance on test set
        acc = validation(test)
        logger.info(f'|Test| Acc:{acc:6.2f}\n')

    print(f"prepare_time_tot:{prepare_time_tot:.2f} process_time_tot:{process_time_tot:.2f}")
    logger.info(f'*Best Dev Result* Acc:{best_acc:6.2f}, Epoch({best_epoch})')
    load_ckpt(BEST)
    model.eval()
    acc = validation(test)
    logger.info(f'*Final Test Result* Acc:{acc:6.2f}')


if __name__ == '__main__':
    main()