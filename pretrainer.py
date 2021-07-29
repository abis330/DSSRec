"""
Script for pre-training the data using following models:
1. S3-Rec (Self-Supervised Sequential Recommender)
    https://arxiv.org/pdf/2008.07873.pdf
2. DSS-Rec (Disentangled Self-Supervised Recommender)
    http://pengcui.thumedialab.com/papers/DisentangledSequentialRecommendation.pdf
3. MIMDSS-Rec (Mutual Information Maximization-based DSS-Rec)
    new loss function combining MIM and DSS

@author: Abinash Sinha
"""

import torch
from torch.utils.data import DataLoader, RandomSampler

import os

from datasets import PretrainDataset
from utils import get_args, get_user_seqs_long_csv, get_item2attribute, check_path, set_seed


def pretrain(args):
    """
    Method to pre-train on data using self-supervision
    :param args:
    :return:
    """
    set_seed(args.seed)
    output_path = os.path.join(args.output_dir, args.loss_type)
    check_path(output_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # get data
    args.data_file = os.path.join(args.data_dir, args.data_name + '.csv')
    item2attribute_file = os.path.join(args.data_dir, args.data_name + '_video2course.csv')
    user_seq, max_item, long_sequence = get_user_seqs_long_csv(args.data_file)
    item2attribute, attribute_size = get_item2attribute(item2attribute_file)
    args.item2attribute = item2attribute

    # checkpoint
    ckp_file = f'{args.model_name}-{args.loss_type}-{args.data_name}-epochs-{args.ckp}.pt'
    args.checkpoint_path = os.path.join(output_path, ckp_file)

    # number of items and mask
    args.item_size = max_item + 2
    args.attribute_size = attribute_size + 1
    args.mask_id = max_item + 1

    # log the arguments
    log_filename = f'{args.model_name}-{args.loss_type}-{args.data_name}.txt'
    args.log_file = os.path.join(output_path, log_filename)
    print(args)
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    if args.loss_type == 'DSS':
        print('Running DSSRec model!')
        from models import DSSRecModel
        from trainers import DSSPretrainer
        model = DSSRecModel(args)
        trainer = DSSPretrainer(model, None, None, None, args)
    elif args.loss_type == 'DSS-2':
        print('Running DSS-2 model!')
        from models import DSSRecModel2
        from trainers import DSSPretrainer
        model = DSSRecModel2(args)
        trainer = DSSPretrainer(model, None, None, None, args)
    else:
        raise ValueError('Invalid loss type!')

    # to resume training from last pre-trained epoch
    if os.path.exists(args.checkpoint_path):
        trainer.load(args.checkpoint_path)
        print(f'Resume training from epoch={args.ckp} for pre-training!')
        init_epoch = int(args.ckp) - 1
    else:
        init_epoch = -1
    for epoch in range(args.pre_epochs):
        if epoch <= init_epoch:
            continue

        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

        trainer.pretrain(epoch, pretrain_dataloader)

        # save checkpoint after execution of each epoch
        ckp = f'{args.model_name}-{args.loss_type}-{args.data_name}-epochs-{epoch+1}.pt'
        checkpoint_path = os.path.join(output_path, ckp)
        trainer.save(checkpoint_path)


if __name__ == '__main__':
    args = get_args()
    pretrain(args)
