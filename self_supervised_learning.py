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


print("self supervised learning start!")

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
if not os.path.isdir("results/probelem{}".format(args.ckpt)):
    os.mkdir("results/probelem{}".format(args.ckpt))
if not os.path.isdir("results/probelem{}/{}".format(args.ckpt, args.folder_name)):
    os.mkdir("results/probelem{}/{}".format(args.ckpt, args.folder_name))
f = open("results/probelem{}/{}/log.txt".format(args.ckpt, args.folder_name), "w")
g = open("results/probelem{}/{}/result.txt".format(args.ckpt, args.folder_name), "w")
tw = csv.writer(g, delimiter='\t')
tw.writerow(['iteration', 'epochs', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'f1_sc'])





# -- pretraining model loading
bertmodel, vocab = get_pytorch_kobert_model(cachedir = "/NasData/home/ksy/Basic/project/project_test/research1/.cache/")
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
pretrain_model = torch.load("model/{}.pt".format(args.pretraining_model))
model.load_state_dict(pretrain_model)
model = nn.DataParallel(model, device_ids = [0, 1, 2, 3]) 


#--data loading
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)
test = nlp.data.TSVDataset("data/{}.txt".format(args.test_data), field_indices =[0, 1], num_discard_samples = 1)
test_dataset = BERTDataset(test, 0, 1, tok, args.max_len, True, False, True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=5)
batch_test= len(test_dataloader)
print("data - train_dataset:{}, test_dataset:{}".format(0, len(test_dataset)));f.write("data - train_dataset:{}, test_dataset:{}".format(0, len(test_dataset))) 
print("dataloader - train_dataloader:{}, test_dataloader:{}".format(0, len(test_dataloader)));f.write("dataloader - train_dataloader:{}, test_dataloader:{}".format(0, len(test_dataloader)))


model.eval()   
 
test_acc, test_loss, targets_list, preds_list = evaluation(0, 0,f, test_dataloader,model, device)
f1_sc = f1(targets_list, preds_list, 0, 0, args)
print("f1_sc:", f1_sc)
test_acc = (test_acc / batch_test)*100
tw.writerow(["-", "-", "-", "-", str(test_acc), str(test_loss), str(f1_sc)])
    
f.close()
g.close()        

