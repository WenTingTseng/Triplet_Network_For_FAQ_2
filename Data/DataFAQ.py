##################產生資料faq.txt################
import sys
import difflib
def get_key (dict, value):
    for k, v in dict.items():  
        if v == value:
            return [k]
train_data_file='traindata.txt'
question_file='北衛_question.txt'
answer_file='北衛_answer.txt'
index_file='北衛_index.txt'
output_file="faq.txt"

fp = open(train_data_file, "r")
train_data = fp.readlines()
fp.close()

fp = open(question_file, "r")
questions = fp.readlines()
fp.close()

fp = open(answer_file, "r")
answers = fp.readlines()
fp.close()

fp = open(index_file, "r")
index = fp.readlines()
fp.close()

idxtoq=dict()
idxtoa=dict()
for i,q in enumerate(questions):
    idxtoq[i]=q.strip('\n')
for i,a in enumerate(answers):
    idxtoa[i]=a.strip('\n')

#寫檔
wp = open(output_file, "w")
k=0
for data in train_data:
    data=data.strip('\n')
    wp.writelines(data+'\n')
    question=data.split()[1]
    print(question)
    wp.writelines("   "+idxtoa[get_key(idxtoq,question)[0]]+'\n')
    k+=1
wp.close()