from sentence_transformers import SentenceTransformer, util
import json
import torch
from transformers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from tqdm import tqdm
import numpy as np 
idf_dir='Data/idf.txt'
faq_dir='Data/faq.txt'
test_dir='Data/test_set.txt'
fqa_len=6331 #faq.txt資料行數除2
test_len=633 #測試資料筆數
once_data_len=500 #跑一次要儲存多少筆Tensor資料
final_data=6330 #最後一筆資料在第幾筆，等於fqa_len減一
output_dir = 'model_cls/'

idf={}
with open(idf_dir,'r') as f:
    while 1:
        line=f.readline()
        if len(line)==0 or line=='\n':
            break 
        line=[float(e) for e in line.split()]
        idf[line[0]]=line[1]

class Sbert(nn.Module):
    def __init__(self,idf):
        super(Sbert, self).__init__()
        self.bert= BertModel.from_pretrained(output_dir)
        self.idf=idf
    def forward(self, in1,in1m,pooling='cls'):
        loss1, a = self.bert(in1, 
                             token_type_ids=None, 
                             attention_mask=in1m)
#################pooling###########################
        if pooling=='idf':
            for i in range(len(in1)):
                for j in range(100):
                    if in1m[i][j]==1:
                        idf_weight=0.0
                        if int( in1[i][j]) in self.idf:
                            idf_weight=math.log(fqa_len/(1+self.idf[int(in1[i][j])]),2)
                        else:
                            idf_weight=math.log(fqa_len/1,2)
                        loss1[i][j]*=idf_weight

            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
            sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
            output_vector1 = sum_embeddings1 / sum_mask1

        if pooling=='avg':
            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
            sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
            output_vector1 = sum_embeddings1 / sum_mask1
        
#[cls]token#
        if pooling=='cls':
            output_vector1=loss1[:, 0, :].float() 
#max#
        if pooling=='max':
            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            loss1[input_mask_expanded1 == 0] = -1e9 
            output_vector1 = torch.max(loss1, 1)[0]

        return output_vector1

tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')#中文 (英文:bert-base-uncased)
model=Sbert(idf)

FAQ=[]

##load faq and testset
with open(faq_dir,"r") as f:
    for i in range(fqa_len):
        line=f.readline()
        line=[e for e in line.split()]
        question="".join(line[1:])
        answer=f.readline()
        FAQ.append([i,int(line[0]),question,answer])
testdata=[]
with open(test_dir,"r") as f:
    for i in range(test_len):
        line=f.readline()
        line=[e for e in line.split()] 
        query="".join(line[1:])
        testdata.append([int(line[0]),query])  
faqvector_ls=[]
k=0
for j in tqdm(range(fqa_len)):
    encoded_dict2=tokenizer.encode_plus(
                FAQ[j][2],                    
                add_special_tokens = True, 
                max_length = 100,          
                pad_to_max_length = True,
                return_attention_mask = True,   
                return_tensors = 'pt',  
                truncation=True   
            )
    faqvector=model(encoded_dict2['input_ids'],encoded_dict2['attention_mask'],'cls')
    #torch.save(faqvector, 'faqvector.pt')
    faqvector_ls.append(faqvector)
    
    if((j%once_data_len==0 and j!=0) or (j==final_data)):         
        torch.save(faqvector_ls, 'Tensor/faqvector'+str(k)+'.pt')
        faqvector_ls=[] 
        k+=1
#torch.save(faqvector, 'faqvector.pt')
# import torch
# faqvector_arr=torch.load('faqvector.pt')
# for i in faqvector_arr.detach().numpy().tolist():
#     input("i")
#     print(i)
# np.save('faqvector_arr.npy',faqvector_arr)