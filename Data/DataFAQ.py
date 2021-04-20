##################產生資料faq.txt################
import sys
train_data_file='traindata.txt'
answer_file='北衛_answer.txt',
index_file='北衛_index.txt'
output_file="faq.txt"

fp = open(train_data_file, "r")
train_data = fp.readlines()
fp.close()

fp = open(answer_file, "r")
answers = fp.readlines()
fp.close()

fp = open(index_file, "r")
index = fp.readlines()
fp.close()

idxtoa=dict()
numtoidx=dict()


for idx,a in zip(index,answers):
    idxtoa[idx]=a

index=set(index)
for num,idx in enumerate(index):
    numtoidx[num]=idx

#寫檔
wp = open(output_file, "w")
k=0
for data in train_data:
    # wp.writelines('\n'+str(k)+'\n')
    wp.writelines(data+'\n')
    data=data.split()
    wp.writelines("   "+idxtoa[numtoidx[int(data[0])]])
    k+=1
print(k)
wp.close()