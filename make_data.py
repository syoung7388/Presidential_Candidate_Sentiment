# -*- coding: cp949 -*
import pickle 
import csv
import argparse
import gluonnlp as nlp


train = nlp.data.TSVDataset("data/movie_train_200k.txt", field_indices =[1, 2], num_discard_samples = 1)


with open("data/movie_train_2k.txt", 'w', encoding='utf-8', newline='') as f:
		tw = csv.writer(f, delimiter='\t')
		tw.writerow(['review', 'label'])		
		for review, label in train[:2000]:
				tw.writerow([review, label])
					
      
      
      

