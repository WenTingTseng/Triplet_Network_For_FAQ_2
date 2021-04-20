##################產生資料traindata.txt################
import sys
question_file='北衛_question.txt'
answer_file='北衛_answer.txt',
index_file='北衛_index.txt'
output_file="traindata.txt"

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

fp = open(question_file, "r")
questions = fp.readlines()
fp.close()

fp = open(answer_file, "r")
answer = fp.readlines()
fp.close()

fp = open(index_file, "r")
index = fp.readlines()
fp.close()

idxls=[]
qtoidx=dict()
idxtonum=dict()
for idx,q in zip(index,questions):
    qtoidx[q]=idx
    idxls.append(idx)
idxls=list(set(idxls))

# index=set(index)
# for num,idx in enumerate(index):
#     idxtonum[idx]=num

numdict=dict()
for key,value in qtoidx.items():
    numdict[key]=idxls.index(value)

Newnumdict=sorted(numdict.items(), key=lambda x:x[1])

# print(len(numdict))
#寫檔
wp = open(output_file, "w")
for data in Newnumdict:
    wp.writelines(str(data[1])+" "+data[0])
wp.close()
 