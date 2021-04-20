##################產生資料train_set.valid_set.txt、test_set.txt、faq_threadQ.txt################
import sys,random

train_data_file='traindata.txt'
output_faq_threadQ_file='faq_threadQ2.txt'
output_train_file="train_set.txt"
output_answer_file="valid_set.txt"
output_index_file="test_set.txt"


questiondata=[]
FAQdata=dict()
with open(train_data_file,'r') as f:
    while 1:
        lines=f.readline()
        if len(lines)==0:
            break
        line=[e for e in lines.split()]
        question="".join(line[1:])
        FAQdata[line[0]]=question
        questiondata.append([int(line[0]),question])
random.shuffle(questiondata)
trainlen=int(len(questiondata)*0.8)
validlen=int(len(questiondata)*0.1)
testlen=int(len(questiondata)*0.1)
train_data=questiondata[:trainlen]
valid_data=questiondata[trainlen:trainlen+validlen]
test_data=questiondata[trainlen+validlen:]

wp = open(output_faq_threadQ_file, "w")
for k,v in FAQdata.items():
    wp.writelines(k+" "+v+'\n')
wp.close()
with open(output_train_file,"w") as f:
    for i in range(trainlen):
        line=str(train_data[i][0])+' '+train_data[i][1]
        f.writelines(line)
        f.writelines("\n")
with open(output_answer_file,"w") as f:
    for i in range(validlen):
        line=str(valid_data[i][0])+' '+valid_data[i][1]
        f.writelines(line)        
        f.writelines("\n")
with open(output_index_file,"w") as f:
    for i in range(testlen):
        line=str(test_data[i][0])+' '+test_data[i][1]
        f.writelines(line)
        f.writelines("\n")