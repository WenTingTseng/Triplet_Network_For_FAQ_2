import torch
from transformers import *
from torch.utils.data import BatchSampler
from torch.utils.data import RandomSampler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import datetime
import os
from model import *
from tqdm import tqdm

faq_threadQ_len=899
train_set_len=5064
valid_set_len=633
idf_file="Data/idf.txt"
faq_threadQ_file="Data/faq_threadQ.txt"
train_file="Data/train_set.txt"
valid_file="Data/valid_set.txt"
output_dir = 'model_cls/'

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
#########set random seed#############
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
###########idf#####################
idf={}
with open(idf_file,'r') as f:
    while 1:
        line=f.readline()
        if len(line)==0 or line=='\n':
            break 
        line=[float(e) for e in line.split()]
        idf[line[0]]=line[1]
#########model################
# tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')#英文
tokenizer=BertTokenizer.from_pretrained('bert-base-chinese')#中文
model=Sbert(idf)
########todevice#########################
device = torch.device("cuda")  # torch.device('cuda') if cuda  is available
model.cuda()                  #use if cuda  is available
##########read file###########

faqset=[]
with open(faq_threadQ_file,"r") as f:
    for i in range(faq_threadQ_len):
        lines=f.readline()      
        line=[e for e in lines.split()]
        faqquestion="".join(line[1:])
        faqset.append([i,faqquestion])

train_data=[]
with open(train_file,"r") as f:
    for i in range(train_set_len):
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        query="".join(line[1:])
        train_data.append([int(line[0]),query])
       

valid_data=[]
with open(valid_file,"r") as f:
    for i in range(valid_set_len):
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        query="".join(line[1:])
        valid_data.append([int(line[0]),query])          

##########enlarge training sample##########

train_input=[]
negative_sum=3
positive_sum=1
for i in range(train_set_len):
    for j in range(positive_sum):
        # print(i)
        # print(train_data[i][1])
        # print(train_data[i][0])
        # print(faqset[train_data[i][0]])
        train_input.append([train_data[i][1],faqset[train_data[i][0]][1],1.0])
    randn=random.sample(list(range(0,train_data[i][0]))+list(range(train_data[i][0]+1,125)),k=negative_sum)
    for j in range(negative_sum):
        train_input.append([train_data[i][1],faqset[randn[j]][1],0.0])

#########valid_input#########################

valid_input=[]
for i in range(valid_set_len):
    valid_input.append([valid_data[i][1],faqset[valid_data[i][0]][1],1.0])
##################encode traindata#################
input1_ids=[]
input1_masks=[]
input2_ids=[]
input2_masks=[]
labels=[]
for i in range(train_set_len*4): 
    encoded_dict1 = tokenizer.encode_plus(
                            train_input[i][0],                    
                            add_special_tokens = True,
                            max_length = 100,          
                            #padding = 'max_length',
                            pad_to_max_length = True, 
                            return_attention_mask = True,   
                            return_tensors = 'pt',
                            truncation=True     
                    )   
    encoded_dict2 = tokenizer.encode_plus(
                            train_input[i][1],                    
                            add_special_tokens = True,
                            max_length = 100,         
                            #padding = 'max_length',
                            pad_to_max_length = True, 
                            return_attention_mask = True,   
                            return_tensors = 'pt',
                            truncation=True  
                    )  
    input1_ids.append(encoded_dict1['input_ids'])
    input1_masks.append(encoded_dict1['attention_mask'])
    input2_ids.append(encoded_dict2['input_ids'])
    input2_masks.append(encoded_dict2['attention_mask'])
    labels.append(torch.tensor([train_input[i][2]]))

input1_ids = torch.cat(input1_ids, dim=0)
input1_masks = torch.cat(input1_masks, dim=0)
input2_ids = torch.cat(input2_ids, dim=0)
input2_masks = torch.cat(input2_masks, dim=0)
labels = torch.cat(labels, dim=0)
######################encode validdata###################

vinput1_ids=[]
vinput1_masks=[]
vinput2_ids=[]
vinput2_masks=[]
vlabels=[]
for i in range(valid_set_len): 
    encoded_dict1 = tokenizer.encode_plus(
                            valid_input[i][0],                    
                            add_special_tokens = True,
                            max_length = 100,          
                            #padding = 'max_length',
                            pad_to_max_length = True, 
                            return_attention_mask = True,   
                            return_tensors = 'pt',
                            truncation=True     
                    )   
    encoded_dict2 = tokenizer.encode_plus(
                            valid_input[i][1],                    
                            add_special_tokens = True,
                            max_length = 100,         
                            #padding = 'max_length',
                            pad_to_max_length = True, 
                            return_attention_mask = True,   
                            return_tensors = 'pt',
                            truncation=True  
                    )    
    vinput1_ids.append(encoded_dict1['input_ids'])
    vinput1_masks.append(encoded_dict1['attention_mask'])
    vinput2_ids.append(encoded_dict2['input_ids'])
    vinput2_masks.append(encoded_dict2['attention_mask'])
    vlabels.append(torch.tensor([valid_input[i][2]]))

vinput1_ids = torch.cat(vinput1_ids, dim=0)
vinput1_masks = torch.cat(vinput1_masks, dim=0)
vinput2_ids = torch.cat(vinput2_ids, dim=0)
vinput2_masks = torch.cat(vinput2_masks, dim=0)
vlabels = torch.cat(vlabels, dim=0)

####################################################
batch_size = 32

train_dataset = TensorDataset(input1_ids,input1_masks,input2_ids,input2_masks,labels)
train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size 
        )

valid_dataset = TensorDataset(vinput1_ids,vinput1_masks,vinput2_ids,vinput2_masks,vlabels)
valid_dataloader = DataLoader(
            valid_dataset,  
            sampler = RandomSampler(valid_dataset), 
            batch_size = batch_size 
        )

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8 
                )

epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

total_t0 = time.time()

#########training#####################
for epoch_i in tqdm(range(0, epochs)):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):

        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step+1, len(train_dataloader), elapsed))

        b_input_ids1 = batch[0].to(device)
        b_input_mask1 = batch[1].to(device)
        b_input_ids2 = batch[2].to(device)
        b_input_mask2 = batch[3].to(device)
        b_label = batch[4].to(device)

        model.zero_grad()        

        loss=model(b_input_ids1,b_input_mask1,b_input_ids2,b_input_mask2,b_label,'cls')
        total_train_loss += loss.item()
        print("total loss:",total_train_loss,"\naverage loss:",total_train_loss/(step+1),"\n--------------------------------------------------------")
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))

    print("")
    print("Running Validation...")
    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0

    for step, batch in enumerate(valid_dataloader):

        vb_input_ids1 = batch[0].to(device)
        vb_input_mask1 = batch[1].to(device)
        vb_input_ids2 = batch[2].to(device)
        vb_input_mask2 = batch[3].to(device)
        vb_label = batch[4].to(device)
        
        with torch.no_grad():        
            vloss=model(vb_input_ids1,vb_input_mask1,vb_input_ids2,vb_input_mask2,vb_label,'cls')
            total_eval_loss += vloss.item()
            print("total loss:",total_eval_loss,"\naverage loss:",total_eval_loss/(step+1),"\n---------------------------------------------")
        
    avg_val_loss = total_eval_loss / len(valid_dataloader)
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)
model_to_save = model.bert.module if hasattr(model, 'module') else model.bert  
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
