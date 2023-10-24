


# calculate the sentence length
import json 
files =  "./datasets/OpenIE4_train_zh.json"
with open(files) as json_file:
    data = json.load(json_file)

ele_num = len(data)


max_length = 0 
min_length = 101
sum_length = 0 

for index, ele in enumerate(data):
    sentence_length = len(ele["sentence_tokenized"])
    sum_length += sentence_length

    if sentence_length > max_length:
        max_length = sentence_length

    if sentence_length < min_length:
        min_length = sentence_length



average_length = sum_length / ele_num

print(f"max_length: {max_length}")
print(f"min_length: {min_length}")
print(f"average_length: {average_length}")
    

