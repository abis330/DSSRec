"""
Script for fine-tuning the pre-trained model and evaluating using whole sample

@author: Abinash Sinha
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import FinetuneDataset
from trainers import FineTrainer
from utils import get_args, EarlyStopping, get_user_seqs_csv, check_path, set_seed, get_item2attribute


def finetune(args):
    set_seed(args.seed)
    output_path = os.path.join(args.output_dir, args.loss_type)
    check_path(output_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = os.path.join(args.data_dir, args.data_name + '.csv')
    item2attribute_file = os.path.join(args.data_dir, args.data_name + '_video2course.csv')
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = \
        get_user_seqs_csv(args.data_file)
    _, attribute_size = get_item2attribute(item2attribute_file)
    args.item_size = max_item + 2
    args.attribute_size = attribute_size + 1
    args.mask_id = max_item + 1

    filename = f'{args.mode}-{args.model_name}-{args.loss_type}-{args.data_name}-pt_{args.ckp}'
    # save model args
    args.log_file = os.path.join(output_path, filename + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    args.checkpoint_path = os.path.join(output_path, filename + '.pt')

    train_dataset = FinetuneDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = FinetuneDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    test_dataset = FinetuneDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    if args.loss_type == 'DSS':
        print('Loaded DSS model!')
        from models import DSSRecModel
        model = DSSRecModel(args)
    elif args.loss_type == 'DSS-2':
        print('Loaded DSS-2 model')
        from models import DSSRecModel2
        model = DSSRecModel2(args)
    else:
        raise ValueError('Invalid loss type!')

    trainer = FineTrainer(model, train_dataloader, eval_dataloader,
                          test_dataloader, args)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)

    else:
        pretrained_path = os.path.join(output_path, f'pretrain-{args.loss_type}-'
                                                    f'{args.data_name}-epochs-{args.ckp}.pt')
        try:
            trainer.load(pretrained_path)
            print(f'Load Checkpoint From {pretrained_path}!')

        except FileNotFoundError:
            print(f'{pretrained_path} Not Found! The Model is same as SASRec')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on NDCG@20
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model, epoch)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=True)

    print(filename)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(filename + '\n')
        f.write(result_info + '\n')


if __name__ == '__main__':
    args = get_args()

    finetune(args)
