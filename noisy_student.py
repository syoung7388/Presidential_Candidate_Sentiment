# -*- coding: cp949 -*

import torch
import torch.nn as nn
import os
from kobert import get_pytorch_kobert_model
import torch
from transformers import BertModel
import gluonnlp as nlp
from kobert import download, get_tokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from kobert import get_tokenizer
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from urllib import request
import pickle 
import csv
import argparse
from module.BERT import BERTDataset, BERTClassifier , get_suyoung_kobert_model
from module.Function import calc_accuracy, training, evaluation, self_training, f1
import json
import random


print("noisy student start ! ")


# parameter
parser = argparse.ArgumentParser(description = 'description')

parser.add_argument('--max_len', type = int, default = 512,  help = 'max_len')
parser.add_argument('--batch_size', type = int, default = 64,  help = 'batch_size')
parser.add_argument('--warmup_ratio', type = float, default = 0.1,  help = 'warmup_ratio')
parser.add_argument('--num_epochs', type = int, default =5,  help = 'num_epochs')
parser.add_argument('--max_grad_norm', type = int, default =1,  help = 'max_grad_norm')
parser.add_argument('--log_interval', type = int, default =200,  help = 'log_interval')
parser.add_argument('--learning_rate', type = float, default =1e-4,  help = 'learning_rate') #5e-5
parser.add_argument('--iteration', type = int, default = 4,  help = 'iteration')
parser.add_argument('--seed', type=int, default=117, help='seed')
parser.add_argument('--gpu', type=str, default='0, 1, 2, 3', help='gpu')
parser.add_argument('--ckpt', type = int, default = 1,  help = 'ckpt')
parser.add_argument('--train_data', type=str, default='movie_train_200k', help='train_data')
parser.add_argument('--unlabel_data', type=str, default='unlabel_18k', help='unlabel_data')
parser.add_argument('--test_data', type=str, default='test_2k', help='test_data')
parser.add_argument('--folder_name', type=str, default='noisy_student_0', help='folder_name')
parser.add_argument('--pretraining_model', type=str, default='200k_model_0', help='pretraining_model')
parser.add_argument('--noisy', type=int, default= 1, help='noisy')
parser.add_argument('--ratio', type=int, default= 1, help='ratio')
args = parser.parse_args()







# -- GPU
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- seed
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -- log, result
if not os.path.isdir("results/problem{}".format(args.ckpt)):
    os.mkdir("results/problem{}".format(args.ckpt))
if not os.path.isdir("results/problem{}/{}".format(args.ckpt, args.folder_name)):
    os.mkdir("results/problem{}/{}".format(args.ckpt, args.folder_name))

f = open("results/problem{}/{}/log.txt".format(args.ckpt, args.folder_name), "w")
g = open("results/problem{}/{}/result.txt".format(args.ckpt, args.folder_name), "w")
tw = csv.writer(g, delimiter='\t')
tw.writerow(['iteration', 'epochs', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'f1_sc'])


# -- train data cnt
ratio = [i for i in range(1, args.iteration+1)] if args.ratio else [args.iteration]*(args.iteration) 



# -- pretraining model loading
bertmodel, vocab = get_pytorch_kobert_model(cachedir = "/NasData/home/ksy/Basic/project/project_test/research1/.cache/")
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
pretrain_model = torch.load("model/{}.pt".format(args.pretraining_model))
model.load_state_dict(pretrain_model)
model = nn.DataParallel(model, device_ids = [0, 1, 2, 3]) 


# -- data
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

train = nlp.data.TSVDataset("data/{}.txt".format(args.train_data), field_indices =[1, 2] if args.train_data == 'movie_train_200k' else [0, 1], num_discard_samples = 1)
#train = train[:100]


ljm = nlp.data.TSVDataset("/NasData/home/ksy/Basic/project/project_test/TextMining/conversion_data/ljm.tsv", field_indices =[2], num_discard_samples = 1)
ysy = nlp.data.TSVDataset("/NasData/home/ksy/Basic/project/project_test/TextMining/conversion_data/ysy.tsv", field_indices =[2], num_discard_samples = 1)
unlabel = ljm[1001:10001] + ysy[1001:10001] # 18K
random.shuffle(unlabel)
unlabel_split = [unlabel[i: i + int(len(unlabel)/args.iteration)] for i in range(0, len(unlabel), int(len(unlabel)/args.iteration))]

"""
unlabel = nlp.data.TSVDataset("data/{}.tsv".format(args.unlabel_data), field_indices =[0], num_discard_samples = 1)
#unlabel = unlabel[:100]
unlabel_split = [unlabel[i: i + int(len(unlabel)/args.iteration)] for i in range(0, len(unlabel), int(len(unlabel)/args.iteration))]
"""

test = nlp.data.TSVDataset("data/{}.txt".format(args.test_data), field_indices =[0, 1], num_discard_samples = 1)
#test = test[:100]
test_dataset = BERTDataset(test, 0, 1, tok, args.max_len, True, False, True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=5)

print("dataset - train:{}, unlabel:{}, test:{}".format(len(train), len(unlabel),len(test_dataset)))
print("dataloader - train:{}, unlabel:{}, test:{}".format("-", len(unlabel_split),len(test_dataloader)))





#bertmodel, vocab =  get_pytorch_kobert_model(cachedir = "/NasData/home/ksy/Basic/project/project_test/research1/.cache/") if not args.noisy else get_suyoung_kobert_model()
accum_pseudo = []
for i in range(args.iteration):		
		f.write("{}/{}: iteration start\n".format(i+1,args.iteration)); print("{}/{}: iteration start".format(i+1,args.iteration))
		
		unlabel_dataset = BERTDataset(unlabel_split[i], 0, 1, tok, args.max_len, True, False, False)
		unlabel_dataloader = torch.utils.data.DataLoader(unlabel_dataset,  batch_size=32 , num_workers=5)    
		pseudo = self_training(unlabel_split[i], unlabel_dataloader, model, device)
		accum_pseudo += pseudo
		tot_train_data = list(train[:500*(i+1)]) + accum_pseudo
		print("tot_train_data:{}".format(len(tot_train_data)));f.write("tot_train_data:{}".format(len(tot_train_data)))
		bertmodel, vocab = get_suyoung_kobert_model()
		model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)	
		model = nn.DataParallel(model, device_ids = [0, 1, 2, 3]) 
		
		new_train_dataset = BERTDataset(tot_train_data, 0, 1, tok,args.max_len, True, False, True)
		new_train_dataloader = torch.utils.data.DataLoader(new_train_dataset, batch_size=args.batch_size, num_workers=5)       
		batch_train, batch_test = len(new_train_dataloader), len(test_dataloader)
		
		no_decay = ['bias', 'LayerNorm.weight']
		optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, no_deprecation_warning=True)
		loss_fn = nn.CrossEntropyLoss()     
		t_total = len(new_train_dataloader) * args.num_epochs
		warmup_step = int(t_total *args.warmup_ratio)
		scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
		
		
		
		ch_acc = []
		for e in range(args.num_epochs):
				model.train()
				f.write("epoch{} start\n".format(e+1))
				train_acc, train_loss = training(e, i, f, new_train_dataloader, optimizer, model, loss_fn, scheduler, device, args)
				train_acc = (train_acc / batch_train)*100
				train_loss = (train_loss.item() / batch_train)
				model.eval()
				f.write("epoch{} test start\n".format(e+1))       
				test_acc, test_loss, targets_list,  preds_list = evaluation(e, i,f, test_dataloader,model, device)
				f1_sc = f1(targets_list, preds_list, i, e, args)
				print("f1_sc:", f1_sc)
				test_acc = (test_acc / batch_test)*100
				tw.writerow([str(i+1), str(e+1), str(train_acc), str(train_loss), str(test_acc), str(test_loss), str(f1_sc)])
				#ch_acc.append(f1_sc)
				torch.save(model.module.state_dict(),"results/problem{}/{}/i={}_e={}_model.pt".format(args.ckpt, args.folder_name, i+1, e+1))
		#best model loading
		model =  BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
		idx = ch_acc.index(max(ch_acc))
		model.load_state_dict(torch.load("results/problem{}/{}/i={}_e={}_model.pt".format(args.ckpt, args.folder_name, i+1, idx+1)))
		model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
		exit()
		

f.close()
g.close()        
        

        

