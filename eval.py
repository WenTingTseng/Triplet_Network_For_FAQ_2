######最後的測試，有加入投票機制######
from sentence_transformers import SentenceTransformer, util
import json
import torch
from transformers import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm 
import sys
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from collections import Counter
#參數設定
faq_len=6331
test_len=633 #測試資料筆數
once_data_len=500
faq_file="Data/faq.txt"
test_file="Data/test_set.txt"
idf_file='Data/idf.txt'
choice_type='cls'
output_log_file='Data/result.txt'
model_dir = 'model_cls/'
output_dir="result.txt"
bert_token='bert-base-chinese'#中文:bert-base-chinese;英文:bert-base-uncased

idf={}
with open(idf_file,'r') as f:
    while 1:
        line=f.readline()
        if len(line)==0 or line=='\n':
            break 
        line=[float(e) for e in line.split()]
        idf[line[0]]=line[1]

class Sbert(nn.Module):
    def __init__(self,idf):
        super(Sbert, self).__init__()
        self.bert= BertModel.from_pretrained(model_dir)
        self.idf=idf
    def forward(self, in1,in1m,pooling='idf'):
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
                            idf_weight=math.log(719/(1+self.idf[int(in1[i][j])]),2)
                        else:
                            idf_weight=math.log(719/1,2)
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

tokenizer=BertTokenizer.from_pretrained(bert_token)#中文
model=Sbert(idf)
FAQ=[]
##load faq and testset
with open(faq_file,"r") as f:
    for i in range(faq_len):
        line=f.readline()
        line=[e for e in line.split()]
        question=" ".join(line[1:])
        answer=f.readline()
        FAQ.append([i,int(line[0]),question,answer])

testdata=[]
gold_ls=[]
with open(test_file,"r") as f:
    for i in range(test_len):
        line=f.readline()
        line=[e for e in line.split()] 
        query="".join(line[-1])
        if(line==[]):
            break
        label=line[0].split(',')
        testdata.append([label,query])  
        gold_ls.append(label)

##make score board        
scoreboard=[]
for i in tqdm(range(test_len)):
    #print(testdata[i][1]) #測試資料的句子
    encoded_dict1=tokenizer.encode_plus(
                testdata[i][1],                    
                add_special_tokens = True, 
                max_length = 100,          
                pad_to_max_length = True,
                return_attention_mask = True,   
                return_tensors = 'pt', 
                truncation=True)
    invector=model(encoded_dict1['input_ids'],encoded_dict1['attention_mask'],choice_type)
    faq_score=[] #存一個Query的排名
    k=0
    for idx,j in tqdm(enumerate(range(0,faq_len,once_data_len))):     
        faqvectorls = torch.load('Tensor/faqvector'+str(idx)+'.pt')
        for faqvector in faqvectorls:
            score=float(torch.cosine_similarity(faqvector,invector))
            # if(k==6330):
            #     break
            faq_score.append([FAQ[k][1],score])
            k+=1
    scoreboard.append(faq_score)#存所有Query的排名

predict_ls=[] #預測結果
##sort each score 以下做投票機制
for i in range(test_len):
    Top5_ls=[]
    scoreboard[i].sort(key=lambda s: s[1],reverse=True)
    Top5=scoreboard[i][:5]#儲存第五名結果
    Top1=scoreboard[i][0][0]#儲存第一名結果
    for t in Top5:
        Top5_ls.append(t[0])
    Top5_dict=dict(Counter(Top5_ls))
    time=max(zip(Top5_dict.values(),Top5_dict.keys()))[0]
    label=max(zip(Top5_dict.values(),Top5_dict.keys()))[1]
    if(time>2):
        predict_ls.append(label)
    else:
        predict_ls.append(Top1)
#以下計算準確率
wp = open(output_dir, "w")
count=0
for p,g,test in zip(predict_ls,gold_ls,testdata):
    #wp.writelines(str(str(p) in g)+str(p)+','+' '.join(g)+'\n')
    if(str(p) in g):
        count+=1
    else: #輸出預測錯誤的資料
       wp.writelines(str(p)+' '+','.join(test[0])+' '+test[1]+'\n')
wp.close()
print("準確率:"+str(count/test_len))    
##評估Precision、Recall、F1
# precision = precision_score(gold_ls, predict_ls, average='macro')
# recall = recall_score(gold_ls, predict_ls, average='macro')
# f1 = f1_score(gold_ls,predict_ls, average='macro')
# acc=accuracy_score(gold_ls, predict_ls)
# ##評估mrr map p@5
# mrr=0.0
# p5=0.0
# map=0.0

# for i in range(test_len):
#     count=0.0
#     ap=0.0
#     target_faq=testdata[i][0] #testdata的label資料
#     first=True
#     for j in range(faq_len):
#         if scoreboard[i][j][0]==target_faq:#第i個Query，第j名次，的label資料，如果等於正確標記
#             count+=1
#             ap+=count/(j+1)
#             if first:
#                 mrr+=1/(j+1)#第一次出現的位置
#                 first=False#出現過後了把機制關掉
#         if j==4:
#             p5+=count/5
#     map+=(ap/count)

# map/=test_len
# p5/=test_len
# mrr/=test_len
# print(map,p5,mrr)
# with open(output_log_file,'w') as f:
#     f.writelines("mAP:"+str(map)+'\n')
#     f.writelines("P@5:"+str(p5)+'\n')
#     f.writelines("MRR:"+str(mrr)+'\n')
#     f.writelines("precision:"+str(precision)+'\n')
#     f.writelines("recall:"+str(recall)+'\n')
#     f.writelines("f1:"+str(f1)+'\n')
#     f.writelines("acc:"+str(acc)+'\n')