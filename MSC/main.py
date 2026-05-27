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
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname agnews_trans_subst_20_charswap_20 --num_classes 20
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname searchsnippets_trans_subst_20_charswap_20 --num_classes 20
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname stackoverflow_trans_subst_20_charswap_20 --num_classes 40
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname biomedical_trans_subst_20_charswap_20 --num_classes 40
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_TS_trans_subst_20_charswap_20 --num_classes 200
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_T_trans_subst_20_charswap_20 --num_classes 200
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname googlenews_S_trans_subst_20_charswap_20 --num_classes 200
# python main.py  --objective MCL --augtype explicit --eta 10 --batch_size 500 --max_iter 3000 --bert mpnet --dataname tweet_trans_subst_20_charswap_20 --num_classes 200

# Experiment results
#--------------------------------------------------------
# agnews
#--------------------------------------------------------
# agnews_charswap_10
# Clustering scores: {'NMI': 0.6512168081609829, 'ARI': 0.6538892124931742, 'AMI': 0.6503215278388938}
# ACC: 0.723
# agnews_charswap_20
# Clustering scores: {'NMI': 0.6236128208319022, 'ARI': 0.5785041513638634, 'AMI': 0.6229116733002337}
# ACC: 0.722
# agnews_trans_subst_10
# Clustering scores: {'NMI': 0.6147405410583947, 'ARI': 0.6089126585950958, 'AMI': 0.6139374605749592}
# ACC: 0.678
# agnews_trans_subst_20
# Clustering scores: {'NMI': 0.6002127509775023, 'ARI': 0.5762758601778278, 'AMI': 0.5994729725118346}
# ACC: 0.580
# agnews_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.5373472641893511, 'ARI': 0.4986713354357723, 'AMI': 0.5364236178449545}
# ACC: 0.610
# agnews_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.5659270377940728, 'ARI': 0.5162030490932187, 'AMI': 0.5651404605797277}
# ACC: 0.523
# agnews_word_deletion_10
# Clustering scores: {'NMI': 0.651083094229322, 'ARI': 0.6532906567332334, 'AMI': 0.6503847729116862}
# ACC: 0.723
# agnews_word_deletion_20
# Clustering scores: {'NMI': 0.6627785807871522, 'ARI': 0.6535410439105677, 'AMI': 0.6621222626644931}
# ACC: 0.674
# agnews_paraphrase
# Clustering scores: {'NMI': 0.646685904954742, 'ARI': 0.6016349836549658, 'AMI': 0.6457058962758295}
# ACC: 0.703
# agnews_RTR
# Clustering scores: {'NMI': 0.6002769717021328, 'ARI': 0.576516744136954, 'AMI': 0.5990193655619252}
# ACC: 0.680

#--------------------------------------------------------
# searchsnippets
#--------------------------------------------------------
# searchsnippets_charswap_10
# Clustering scores: {'NMI': 0.7682018986341568, 'ARI': 0.7212086766247124, 'AMI': 0.7654950846620708}
# ACC: 0.823
# searchsnippets_charswap_20
# Clustering scores: {'NMI': 0.7628528003250029, 'ARI': 0.7056917263369812, 'AMI': 0.7623404414588785}
# ACC: 0.813
# searchsnippets_trans_subst_10
# Clustering scores: {'NMI': 0.8005228920063874, 'ARI': 0.6707542165350782, 'AMI': 0.800035532822776}
# ACC: 0.699
# searchsnippets_trans_subst_20
# Clustering scores: {'NMI': 0.6948279061410721, 'ARI': 0.5213898085268819, 'AMI': 0.6941167989054757}
# ACC: 0.602
# searchsnippets_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.7746596434671121, 'ARI': 0.6452254064023889, 'AMI': 0.7741192527855731}
# ACC: 0.664
# searchsnippets_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.6897509614692808, 'ARI': 0.5093740057897097, 'AMI': 0.6890264914882847}
# ACC: 0.591
# searchsnippets_word_deletion_10
# Clustering scores: {'NMI': 0.8308392871301405, 'ARI': 0.7299740486769725, 'AMI': 0.8304204961587054}
# ACC: 0.723
# searchsnippets_word_deletion_20
# Clustering scores: {'NMI': 0.7982514017548609, 'ARI': 0.6827077547933752, 'AMI': 0.7977540457599952}
# ACC: 0.689
# searchsnippets_paraphrase
# Clustering scores: {'NMI': 0.8310038871214061, 'ARI': 0.7270590696626422, 'AMI': 0.8302347094735435}
# ACC: 0.769
# searchsnippets_RTR
# Clustering scores: {'NMI': 0.6953608170445933, 'ARI': 0.521895810665986, 'AMI': 0.6947581655247315}
# ACC: 0.602

#------------------------------------------------------
# stackoverflow
#------------------------------------------------------
# stackoverflow_charswap_10
# Clustering scores: {'NMI': 0.8186955482530887, 'ARI': 0.6965189100822127, 'AMI': 0.8184936313574645}
# ACC: 0.739
# stackoverflow_charswap_20
# Clustering scores: {'NMI': 0.7922069396700084, 'ARI': 0.6816117561999052, 'AMI': 0.7910083222054566}
# ACC: 0.694
# stackoverflow_trans_subst_10
# Clustering scores: {'NMI': 0.8127600560541476, 'ARI': 0.6982731874338136, 'AMI': 0.8166182700322388}
# ACC: 0.718
# stackoverflow_trans_subst_20
# Clustering scores: {'NMI': 0.802228839734804, 'ARI': 0.6601583747567082, 'AMI': 0.8010870976460778}
# ACC: 0.685
# stackoverflow_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.8045266556822951, 'ARI': 0.6775463851449521, 'AMI': 0.8033996533003412}
# ACC: 0.697
# stackoverflow_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.7829400189493211, 'ARI': 0.674196318786615, 'AMI': 0.7816951508684783}
# ACC: 0.686
# stackoverflow_word_deletion_10
# Clustering scores: {'NMI': 0.819215165341575, 'ARI': 0.6972382447692623, 'AMI': 0.8181657385383331}
# ACC: 0.710
# stackoverflow_word_deletion_20
# Clustering scores: {'NMI': 0.7853617158543855, 'ARI': 0.6629548077759699, 'AMI': 0.7841321140012822}
# ACC: 0.677
# stackoverflow_paraphrase
# Clustering scores: {'NMI': 0.8047340769275242, 'ARI': 0.6842043763412843, 'AMI': 0.805598069047828}
# ACC: 0.693
# stackoverflow_RTR
# Clustering scores: {'NMI': 0.8044055872460642, 'ARI': 0.6718843569426868, 'AMI': 0.8031397956127074}
#  ACC: 0.707

#------------------------------------------------------
# biomedical
#------------------------------------------------------
# biomedical_charswap_10
# Clustering scores: {'NMI': 0.596284206688049, 'ARI': 0.4324153372528755, 'AMI': 0.5944243944340479}
# ACC: 0.551
# biomedical_charswap_20
# Clustering scores: {'NMI': 0.5071684599805334, 'ARI': 0.35583333467180217, 'AMI': 0.5042777269880098}
# ACC: 0.502
# biomedical_trans_subst_10
# Clustering scores: {'NMI': 0.5952755164926614, 'ARI': 0.4322677435799984, 'AMI': 0.5931046445200156}
# ACC: 0.517
# biomedical_trans_subst_20
# Clustering scores: {'NMI': 0.5722104162754512, 'ARI': 0.4020896620119041, 'AMI': 0.5697722191224053}
# ACC: 0.495
# biomedical_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.5808354060582414, 'ARI': 0.41504158612125597, 'AMI': 0.5784494936912318}
# ACC: 0.502
# biomedical_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.5566531063660918, 'ARI': 0.3885506023936258, 'AMI': 0.5541302753369759}
# ACC: 0.491
# biomedical_word_deletion_10
# Clustering scores: {'NMI': 0.5959417744833315, 'ARI': 0.4323779118401326, 'AMI': 0.5936470834618054}
# ACC: 0.513
# biomedical_word_deletion_20
# Clustering scores: {'NMI': 0.5829224201080376, 'ARI': 0.4155378431791824, 'AMI': 0.5805466801127633}
# ACC: 0.506
# biomedical_paraphrase
# Clustering scores: {'NMI': 0.5027102348641884, 'ARI': 0.34809521183460373, 'AMI': 0.4997919057542187}
# ACC: 0.499
# biomedical_RTR
# Clustering scores: {'NMI': 0.5006905378439627, 'ARI': 0.35327446064955052, 'AMI': 0.506770277185349}
#  ACC: 0.499

#----------------------------------------------------
# googlenews_TS
#----------------------------------------------------
# googlenews_TS_charswap_10
# Clustering scores: {'NMI': 0.9322091269409986, 'ARI': 0.6702412149789483, 'AMI': 0.932250166983221}
# ACC: 0.862
# googlenews_TS_charswap_20
# Clustering scores: {'NMI': 0.907909296826577, 'ARI': 0.620911521668103, 'AMI': 0.8868064345205383}
# ACC: 0.861
# googlenews_TS_trans_subst_10
# Clustering scores: {'NMI': 0.9350458398160238, 'ARI': 0.6762816338673185, 'AMI': 0.930697535024389}
# ACC: 0.868
# googlenews_TS_trans_subst_20
# Clustering scores: {'NMI': 0.9084427461896276, 'ARI': 0.6146224325074722, 'AMI': 0.8875502783816209}
# ACC: 0.867
# googlenews_TS_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.9081668952785664, 'ARI': 0.6127155572888429, 'AMI': 0.8871725850665533}
# ACC: 0.868
# googlenews_TS_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.9118338234838685, 'ARI': 0.6384537860481022, 'AMI': 0.8918709740874192}
# ACC: 0.887
# googlenews_TS_word_deletion_10
# Clustering scores: {'NMI': 0.9045318247146705, 'ARI': 0.6094251481219661, 'AMI': 0.882728871542874}
# ACC: 0.862
# googlenews_TS_word_deletion_20
# Clustering scores: {'NMI': 0.9039688574166942, 'ARI': 0.60592519529331, 'AMI': 0.8818940972217734}
# ACC: 0.864
# googlenews_TS_paraphrase
# Clustering scores: {'NMI': 0.9250741999371321, 'ARI': 0.654402461272981, 'AMI': 0.9242717976916563}
# ACC: 0.870
# googlenews_TS_RTR
# Clustering scores: {'NMI': 0.9056112360151777, 'ARI': 0.6018181564491667, 'AMI': 0.90359967969464}
#  ACC: 0.859

#-----------------------------------------------------
# googlenews_T
#-----------------------------------------------------
# googlenews_T_charswap_10
# Clustering scores: {'NMI': 0.9387965600641965, 'ARI': 0.6713392125939165, 'AMI': 0.9370918021350942}
# ACC: 0.794
# googlenews_T_charswap_20
# Clustering scores: {'NMI': 0.8698948124336091, 'ARI': 0.6442976508299262, 'AMI': 0.840195502168422}
# ACC: 0.728
# googlenews_T_trans_subst_10
# Clustering scores: {'NMI': 0.9403416546739158, 'ARI': 0.6881241652810801, 'AMI': 0.9368856627399857}
# ACC: 0.798
# googlenews_T_trans_subst_20
# Clustering scores: {'NMI': 0.875329939094179, 'ARI': 0.5719578904223482, 'AMI': 0.8468259012608842}
# ACC: 0.738
# googlenews_T_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.8722622440121687, 'ARI': 0.5586115353339112, 'AMI': 0.842899068617439}
# ACC: 0.735
# googlenews_T_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.8760170259458063, 'ARI': 0.5687036946851867, 'AMI': 0.8476196613013831}
# ACC: 0.737
# googlenews_T_word_deletion_10
# Clustering scores: {'NMI': 0.8700376911541976, 'ARI': 0.5504869378898434, 'AMI': 0.8401423535383559}
# ACC: 0.734
# googlenews_T_word_deletion_20
# Clustering scores: {'NMI': 0.8720335402422169, 'ARI': 0.5466808533364999, 'AMI': 0.8428215042823438}
# ACC: 0.729
# googlenews_T_paraphrase
# Clustering scores: {'NMI': 0.9315415965821756, 'ARI': 0.6622364847034394, 'AMI': 0.9309033396227189}
# ACC: 0.786
# googlenews_T_RTR
#  Clustering scores: {'NMI': 0.9306449278146897, 'ARI': 0.6781386845522906, 'AMI': 0.9279519626470636}
#  ACC: 0.792

#---------------------------------------------------
# googlenews_S
#---------------------------------------------------
# googlenews_S_charswap_10
# Clustering scores: {'NMI': 0.9282677174395707, 'ARI': 0.6860958170823335, 'AMI': 0.9256931959436312}
# ACC: 0.835
# googlenews_S_charswap_20
# Clustering scores: {'NMI': 0.879905594439137, 'ARI': 0.5803591989937358, 'AMI': 0.8529022825964462}
# ACC: 0.811
# googlenews_S_trans_subst_10
# Clustering scores: {'NMI': 0.9257343548610245, 'ARI': 0.6713507028328816, 'AMI': 0.9250584344732646}
# ACC: 0.834
# googlenews_S_trans_subst_20
# Clustering scores: {'NMI': 0.8799589285513131, 'ARI': 0.5840282215538947, 'AMI': 0.8530682493819511}
# ACC: 0.818
# googlenews_S_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.8765552624012111, 'ARI': 0.5783331969440737, 'AMI': 0.8489536628743973}
# ACC: 0.804
# googlenews_S_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.8794948752791147, 'ARI': 0.5831224137471124, 'AMI': 0.8525330599179048}
# ACC: 0.805
# googlenews_S_word_deletion_10
# Clustering scores: {'NMI': 0.8780156440095844, 'ARI': 0.5808040791665396, 'AMI': 0.8506403472705425}
# ACC: 0.793
# googlenews_S_word_deletion_20
# Clustering scores: {'NMI': 0.8750915028189922, 'ARI': 0.5603439977441248, 'AMI': 0.846196840658322}
# ACC: 0.792
# googlenews_S_paraphrase
# Clustering scores: {'NMI': 0.9263667329068695, 'ARI': 0.6785339677431183, 'AMI': 0.9241303784728353}
# ACC: 0.828
# googlenews_S_RTR
# Clustering scores: {'NMI': 0.9190624785541438, 'ARI': 0.678739989789897, 'AMI': 0.9120306534567589}
#  ACC: 0.830

#----------------------------------------------------
# tweet
#----------------------------------------------------
# tweet_charswap_10
# Clustering scores: {'NMI': 0.9228614783589228, 'ARI': 0.55350088004186645, 'AMI': 0.9194923636060495}
# ACC: 0.772
# tweet_charswap_20
# Clustering scores: {'NMI': 0.8299495718434267, 'ARI': 0.5044271943902503, 'AMI': 0.838812589559915}
# ACC: 0.764
# tweet_trans_subst_10
# Clustering scores: {'NMI': 0.9504168688934822, 'ARI': 0.5193032364211924, 'AMI': 0.9461582601344674}
# ACC: 0.783
# tweet_trans_subst_20
# Clustering scores: {'NMI': 0.9412452037983531, 'ARI': 0.5164339783741536, 'AMI': 0.9416589009026735}
# ACC: 0.754
# tweet_trans_subst_10_charswap_10
# Clustering scores: {'NMI': 0.8825203720715483, 'ARI': 0.5011200707370223, 'AMI': 0.884609315440846}
# ACC: 0.763
# tweet_trans_subst_20_charswap_20
# Clustering scores: {'NMI': 0.8737648251871519, 'ARI': 0.5011883234412832, 'AMI': 0.874831433426362}
# ACC: 0.763
# tweet_word_deletion_10
# Clustering scores: {'NMI': 0.8684747881159128, 'ARI': 0.5018260969558031, 'AMI': 0.8686791309064644}
# ACC: 0.767
# tweet_word_deletion_20
# Clustering scores: {'NMI': 0.8597957913226356, 'ARI': 0.4917154796741952, 'AMI': 0.8585228968837957}
# ACC: 0.764
# tweet_paraphrase
# Clustering scores: {'NMI': 0.9522894487300173, 'ARI': 0.58188105297883457, 'AMI': 0.9482869324157023}
# ACC: 0.812
# tweet_RTR
# Clustering scores: {'NMI': 0.9309107266098093, 'ARI': 0.52794761793898415, 'AMI': 0.9321479790932121}
#  ACC: 0.789
