import pickle 
import csv
import argparse



parser = argparse.ArgumentParser(description = 'return Tsv')
parser.add_argument('--name', type= str, default = 'ysy', help = 'name')
args = parser.parse_args()


f = open("/NasData/home/ksy/Basic/project/project_test/TextMining/textmining_data/"+args.name+".p", "rb")
data = pickle.load(f)


with open("/NasData/home/ksy/Basic/project/project_test/TextMining/conversion_data/"+args.name+".tsv", 'w', encoding='utf-8', newline='') as f:
    tw = csv.writer(f, delimiter='\t')
    tw.writerow(['name', 'date', 'comment'])
    
    for nickname, date, comment in data:
        tw.writerow([nickname, date, comment])
      
      
      
      