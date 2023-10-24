import json
import pickle
from transformers import BasicTokenizer

tokenizer = BasicTokenizer(do_lower_case=False)

Original_Data_File = './datasets/Re-OIE2016-Spanish-Clean.json' 
with open(Original_Data_File) as json_file:
    js_file = json.load(json_file)

evaluate_file = './evaluate/Re-OIE2016-Spanish-Clean-Binary.json'
evaluate_json_file = open(evaluate_file, "a")


def find_subarray_start(a, b):
    if len(a) > len(b):
        return -1  # a的长度大于b，a不可能是b的子数组

    start_index = -1  # a在b中的起始位置，默认为-1表示没有找到

    for i in range(len(b)):

        if b[i] == a[0]:  # 找到a的第一个元素在b中的索引
            if b[i:i+len(a)] == a:  # 比较a和b中相应位置的元素
                start_index = i  # a是b的子数组，更新起始位置
                break

    return start_index

# sentence_list = []
data_dict = {}
negative = 0
tuple_sum = 0
for index, (sentence, labels) in enumerate(js_file.items()):
    labels_list = []
    sen_toks = tokenizer.tokenize(sentence) 
       
    for element in labels:
        tuple_dict = {"arg0": "", "arg0_index": [], "pred": "", "pred_index": [],\
                "arg1": "", "arg1_index": [], "arg2": "", "arg2_index": [], \
                "arg3": "", "arg3_index": [], "loc": "", "loc_index": [], "temp": "", "temp_index": [], "context": "", "context_index": []\
            }
        
        tuple_sum += 1
        sub_item = element["arg0"].strip()
        rel_item = element["pred"].strip()
        obj_item = element["arg1"].strip()


        sub_item_toks = tokenizer.tokenize(sub_item)
        rel_item_toks = tokenizer.tokenize(rel_item)
        obj_item_toks = tokenizer.tokenize(obj_item)


        sub_start = find_subarray_start(sub_item_toks, sen_toks)
        #sub_start = sentence.index(sub_item)
        tuple_dict["arg0"] = sub_item
        sub_end = sub_start + len(sub_item_toks) - 1
        #sub_end = sub_start + len(sub_item.split(' ')) - 1
        tuple_dict["arg0_index"] = [sub_start, sub_end]

        rel_start = find_subarray_start(rel_item_toks, sen_toks)
        tuple_dict["pred"] = rel_item
        rel_end = rel_start + len(rel_item_toks) - 1
        tuple_dict["pred_index"] = [rel_start, rel_end]


        try:
            obj_start = find_subarray_start(obj_item_toks, sen_toks)
        except:
            tuple_dict["arg1_index"] = []

        if sub_start == -1 or rel_start == -1 or obj_start == -1:
            negative += 1
            
        tuple_dict["arg1"] = obj_item

        if obj_item != "":
            obj_end = obj_start + len(obj_item_toks) - 1
            tuple_dict["arg1_index"] = [obj_start, obj_end]

        labels_list.append(tuple_dict)



    data_dict[sentence] = labels_list
    

print(tuple_sum)
print(negative)
print(negative/tuple_sum)
exit()


json.dump(data_dict, evaluate_json_file, ensure_ascii=False)







