"""
Copyright https://www.ynu.edu.cn/ or its affiliates school. All Rights Reserved

Author: Kongqiang Wang (wangkongqiang60@gmail.com)
Date: 05/27/2026
"""

import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from models.Transformers import SCCLBert
import dataloader.dataloader as dataloader
from my_training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers
from utils.logger import setup_path, set_global_random_seed
from utils.optimizer import get_optimizer, get_bert
import numpy as np

def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    # train_loader = dataloader.explict_augmentation_loader(args) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)

    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)
    train_loader = dataloader.explict_augmentation_loader(args,tokenizer) if args.augtype == "explicit" else dataloader.virtual_augmentation_loader(args)
    # initialize cluster centers
    cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length)
    
    model = SCCLBert(bert, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha,num=args.num_classes)
    model = model.cuda()

    # optimizer 
    optimizer,optimizer1 = get_optimizer(model, args)
    
    trainer = SCCLvTrainer(model, tokenizer, optimizer,optimizer1, train_loader, args)
    trainer.train()
    
    return None

def get_args(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 or -1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')
    
    parser.add_argument('--bert', type=str, default='distilbert', help="")
    parser.add_argument('--use_pretrain', type=str, default='SBERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    # Dataset
    parser.add_argument('--datapath', type=str, default='../datasets/textdatasets/augmentation/')
    parser.add_argument('--dataname', type=str, default='searchsnippets_charswap_10', help="")
    parser.add_argument('--num_classes', type=int, default=20, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=1e-5, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=1000)
    # contrastive learning
    parser.add_argument('--objective', type=str, default='contrastive')
    parser.add_argument('--augtype', type=str, default='virtual', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--temperature', type=float, default=0.5, help="temperature required by contrastive loss")
    parser.add_argument('--topk', type=int, default=500)
    parser.add_argument('--eta', type=float, default=1, help="")
    # Clustering
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args

if __name__ == '__main__':
    import subprocess
    l = [0.5]
    for i in l:
        print("temperature parameter",i)
        args = get_args(sys.argv[1:])
        args.temperature=i
        if args.train_instance == "sagemaker":
            run(args)
            subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
        else:
            run(args)

# Switch to the current path
# cd /mnt/c/Users/wangkongqiang/PycharmProjects/pythonProject/MSC/

# Activate the virtual environment
# conda activate MSC

# Install the correct version of faiss using Conda
# conda install -c pytorch faiss-gpu==1.7.0
# This will automatically solve the problem of _swigfaiss because the conda package contains compiled dependencies.

# Run the clustering experiment script
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname agnews_trans_subst_20_charswap_20 --num_classes 20
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname searchsnippets_trans_subst_20_charswap_20 --num_classes 20
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname stackoverflow_trans_subst_20_charswap_20 --num_classes 40
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname biomedical_trans_subst_20_charswap_20 --num_classes 40
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_TS_trans_subst_20_charswap_20 --num_classes 200
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_T_trans_subst_20_charswap_20 --num_classes 200
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_S_trans_subst_20_charswap_20 --num_classes 200
# python main.py  --objective MSC --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname tweet_trans_subst_20_charswap_20 --num_classes 200
