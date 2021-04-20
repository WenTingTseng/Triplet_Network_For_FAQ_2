import torch
from transformers import BertTokenizer 
import math
import sys

data_len=6331
data_file='traindata.txt'
faq_file="faq.txt"
output_file='idf.txt'
# tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')#英文
tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')#中文

fp = open(data_file, "r")
questions = fp.readlines()
fp.close()

FAQ=[]
with open(faq_file,"r") as f:
    for i in range(len(questions)):
        line=f.readline()
        line=[e for e in line.split()]
        print(line)
        print(line==[])
        if(line==[]):
            continue
        question=" ".join(line[1])
        answer=f.readline()
        tokens1 = tokenizer.tokenize(question)
        ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        tokens2 = tokenizer.tokenize(answer)
        ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        FAQ.append(ids1+ids2)

term={}
for i in range(data_len):
    print(i)
    for e in FAQ[i]:
        if e not in term:
            term[e]=0.0
    for j in range(len(FAQ[i])):
            if FAQ[i][j] not in FAQ[i][:j]:
                term[FAQ[i][j]]+=1.0   

with open(output_file,'w') as f:
    for i in term:
        f.writelines(str(i)+' '+str(term[i])+'\n')
