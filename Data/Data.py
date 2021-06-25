##################產生資料traindata.txt################
import sys
def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

fp = open('北衛_index.txt', "r")
index = fp.readlines()
fp.close()

idxls=list(set(index))
wp = open("IdxToNum.txt", "w")
for i,idx in enumerate(idxls):
    wp.writelines(str(i)+" "+idx)
wp.close()

fp = open('北衛_question.txt', "r")
questions = fp.readlines()
fp.close()

fp = open('北衛_index.txt', "r")
index = fp.readlines()
fp.close()

idxtonum={}
fp = open('IdxToNum.txt', "r")
for line in iter(fp):
    idxtonum[line.split()[1].strip('\n')]=int(line.split()[0])

questiontonum=dict()
for q,idx in zip(questions,index):
    questiontonum[q]=idxtonum[idx.strip('\n')]
for q,idx in zip(questions,index):
    if(idx.strip('\n')=="QA-1-210"):
        questiontonum[q]=idxtonum[idx.strip('\n')]

Newnumdict=sorted(questiontonum.items(), key=lambda x:x[1])

wp = open("traindata.txt", "w")
for data in Newnumdict:
    wp.writelines(str(data[1])+" "+data[0])
wp.close()
