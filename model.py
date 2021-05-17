import torch
from transformers import *
import torch.nn as nn
import torch.nn.functional as F
import math
output_dir
class Sbert(nn.Module):
    def __init__(self,idf):
        super(Sbert, self).__init__()
        self.bert= BertModel.from_pretrained('bert-base-chinese')
        self.loss_function=nn.MSELoss()
        #self.loss_function=nn.CrossEntropyLoss()
        self.idf=idf
    def forward(self, in1,in1m,in2,in2m,label,pooling='idf'):
        loss1, a = self.bert(in1, 
                             token_type_ids=None, 
                             attention_mask=in1m)
        loss2, b = self.bert(in2, 
                             token_type_ids=None, 
                        attention_mask=in2m)

#################pooling###########################
#average#
  
        if pooling=='idf':
            for i in range(len(in1)):
                for j in range(100):
                    if in1m[i][j]==1:
                        idf_weight=0.0
                        if int( in1[i][j]) in self.idf:
                            idf_weight=math.log(5821/(1+self.idf[int(in1[i][j])]),2)
                        else:
                            idf_weight=math.log(5821/1,2)
                        loss1[i][j]*=idf_weight
                    if in2m[i][j]==1:
                        idf_weight=0.0
                        if  int(in2[i][j]) in self.idf:
                            idf_weight=math.log(5821/(1+self.idf[int(in2[i][j])]),2)
                        else:
                            idf_weight=math.log(5821/1,2)
                        loss2[i][j]*=idf_weight

            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
            sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
            output_vector1 = sum_embeddings1 / sum_mask1

            input_mask_expanded2 = in2m.unsqueeze(-1).expand(loss2.size()).float()
            sum_embeddings2 = torch.sum(loss2 * input_mask_expanded2, 1)
            sum_mask2 = torch.clamp(input_mask_expanded2.sum(1), min=1e-9)
            output_vector2 = sum_embeddings2 / sum_mask2
#avg#
        if pooling=='avg':
            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            sum_embeddings1 = torch.sum(loss1 * input_mask_expanded1, 1)
            sum_mask1 = torch.clamp(input_mask_expanded1.sum(1), min=1e-9)
            output_vector1 = sum_embeddings1 / sum_mask1

            input_mask_expanded2 = in2m.unsqueeze(-1).expand(loss2.size()).float()
            sum_embeddings2 = torch.sum(loss2 * input_mask_expanded2, 1)
            sum_mask2 = torch.clamp(input_mask_expanded2.sum(1), min=1e-9)
            output_vector2 = sum_embeddings2 / sum_mask2

        
#[cls]token#
        if pooling=='cls':
            output_vector1=loss1[:, 0, :].float() 
            output_vector2=loss2[:, 0, :].float() 
#max#
        if pooling=='max':
            input_mask_expanded1 = in1m.unsqueeze(-1).expand(loss1.size()).float()
            loss1[input_mask_expanded1 == 0] = -1e9 
            output_vector1 = torch.max(loss1, 1)[0]

            input_mask_expanded2 = in2m.unsqueeze(-1).expand(loss2.size()).float()
            loss2[input_mask_expanded2 == 0] = -1e9 
            output_vector2 = torch.max(loss2, 1)[0]           
        
#########cosine sim######################
        
        output=torch.cosine_similarity(output_vector1,output_vector2)
        output=self.loss_function(output,label)
        return output
