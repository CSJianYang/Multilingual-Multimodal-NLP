from requests import post
import json
import random
from tqdm import tqdm
import time
import os
from fuzzywuzzy import fuzz
# from transformers import BertTokenizer
from transformers import BasicTokenizer
import langid
import re


random.seed(1)
MATCH_SIMILARITY_THRESHOLD = 95
tokenizer = BasicTokenizer(do_lower_case=False)




LANG_LIST = ["simplified Chinese", "German", "Spanish", "Portuguese","Arabic"]
LANGUAGE = "Arabic" ###
LAST_INDEX = 0 ###
New_Datas_File = './datasets/new_dataset_ar.json' ###
# New_Datas_File_LIST = ['new_dataset_zh.json', 'new_dataset_de.json', 'new_dataset_es.json', 'new_dataset_pt.json']


headers = {"Authorization": "keys",
           "Content-Type": "application/json"} ### aaaa






url = "https://api.openai-proxy.com/v1/chat/completions"
new_json_file = open(New_Datas_File, "a")

Original_Data_File = './datasets/structured_data.json'
with open(Original_Data_File) as json_file:
    js_file = json.load(json_file)

random_selection_data = random.sample(js_file, 1000000)[LAST_INDEX:]
#random_selection_data = random.sample(js_file, 200) # Debugging..............



ALL_MATCHED_SAMPLE_NUM = 0
start = time.time()
def detect_language(text):
    language, confidence = langid.classify(text)
    return language, confidence

def remove_chinese_spaces(text):
    # 删除中文字符间的空格
    text = re.sub(r'([\u4e00-\u9fff])(\s+)([\u4e00-\u9fff])', r'\1\3', text)

    # 删除中文字符和英文字符之间的空格
    text = re.sub(r'([\u4e00-\u9fff])(\s+)([a-zA-Z])', r'\1\3', text, flags=re.UNICODE)
    text = re.sub(r'([a-zA-Z])(\s+)([\u4e00-\u9fff])', r'\1\3', text, flags=re.UNICODE)

    # 删除中文字符和数字、特殊符号之间的空格
    text = re.sub(r'([\u4e00-\u9fff])(\s+)([0-9!@#$%^&*()_\-+=\\|[\]{};:\'",.<>/?])', r'\1\3', text, flags=re.UNICODE)
    text = re.sub(r'([0-9!@#$%^&*()_\-+=\\|[\]{};:\'",.<>/?])(\s+)([\u4e00-\u9fff])', r'\1\3', text, flags=re.UNICODE)

    # 删除标点符号与中文字符之间的空格
    text = re.sub(r'([^\u4e00-\u9fff\s])(\s+)([\u4e00-\u9fff])', r'\1\3', text)

    return text

def find_max_similar_substring(text, pattern):
    n = len(text)
    m = len(pattern)
    max_similarity = MATCH_SIMILARITY_THRESHOLD # threshold==95
    best_match = ""
    best_match_start_index = -1

    for i in range(n - m + 1):
        substring = text[i:i+m]
        similarity = fuzz.token_set_ratio(substring, pattern)
        if similarity >= max_similarity:
            max_similarity = similarity
            best_match = substring # substring of text
            best_match_start_index = i

    if max_similarity >= 95:
        print(f"max_similarity: {max_similarity}")

    return best_match, best_match_start_index

def post_process_sentence(translated_sentence: str):

    while "“" in translated_sentence:
        translated_sentence = translated_sentence.replace("“", "\"")
        translated_sentence = translated_sentence.replace("”", "\"")

    if LANGUAGE == "simplified Chinese":
        translated_sentence = remove_chinese_spaces(translated_sentence)
    else:
        translated_sentence.replace("，", ",")

    return translated_sentence

def post_process_triple(ele_list: list):
    if len(ele_list) > 2:
        triple_ele = ''.join(ele_list[1:]).strip()
    else:
        triple_ele = ele_list[1].strip()

    # empty element
    if triple_ele == "":
        return triple_ele

    if '.' in triple_ele[-1] or '。' in triple_ele[-1]:
        triple_ele = triple_ele[:-1]

    while "“" in triple_ele:
        triple_ele = triple_ele.replace("“", "\"")
        triple_ele = triple_ele.replace("”", "\"")

    if '(' in triple_ele:
        triple_ele = triple_ele.split('(')[0].strip()
    if '（' in triple_ele:
        triple_ele = triple_ele.split('（')[0].strip()

    # delete Chinsese space
    if LANGUAGE == "simplified Chinese":
        triple_ele = remove_chinese_spaces(triple_ele)
    else:
        triple_ele.replace("，", ",")

    return triple_ele


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

def try_correcting_triple(ele:str, sentence:str):
    if '"' in ele:  # delete punctuation ""
        ele = ele.replace('"', "")
    if "'" in ele:  # delete punctuation ''
        ele = ele.replace("'", "")
    # find_max_similar_substring
    result, start_index = find_max_similar_substring(sentence, ele)
    return result, start_index

for Index, cur_data in tqdm(enumerate(random_selection_data, LAST_INDEX)):
    print("\n####################################################################################################################")
    print(f"\nCurrent Sample Index: {Index}")
    print(f"Valid Sample Number: {ALL_MATCHED_SAMPLE_NUM}")
    if ALL_MATCHED_SAMPLE_NUM == 2000:
        break

    cur_sentence = cur_data['sentence'].strip()
    print('\nOriginal Sentence:\n' + cur_sentence)
    # print("\n")

    subject_list = []
    relation_list = []
    obejct_list = []

    valid_sample_data = {
        "sentence": "",
        "sentence_tokenized": [],
        "tuples": []
    }


    has_Empty_ELement = False
    is_Matched = True
    is_Corrected_Triple_Generated = True

    confidence_score = []

    for cur_tuple in cur_data['tuples']:
        cur_sub = cur_tuple['arg0'].strip()
        cur_rel = cur_tuple['relation'].strip()
        cur_obj = ""
        if not cur_tuple['args']:
            has_Empty_ELement = True
            break
        else:
            cur_obj = cur_tuple['args'][0].strip() # List: cur_tuple['args']    ["aaa","bb"]

        if cur_sub == "" or cur_rel == "" or cur_obj == "":
            print(f"There is empty element in cur_sub: {cur_sub} or cur_rel: {cur_rel} or cur_obj: {cur_obj}")
            has_Empty_ELement = True
            break

        subject_list.append(cur_sub)
        relation_list.append(cur_rel)
        obejct_list.append(cur_obj) # [[a,b,c],[a,b],[a]] ["a,b,c", "a,b","a"]
        confidence_score.append(cur_tuple['score'])

    if has_Empty_ELement:
        print("### Empty Elements in Original Triple. Discarded! ###")
        continue

    if not (len(subject_list) == len(relation_list) == len(obejct_list)) or len(subject_list) == 0 or len(relation_list) == 0 or len(obejct_list) == 0:
        print("### Bad data in raw triple. Discarded！ ###")
        continue
############################# Sentence Req #####################################
    sen_data_req = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": \
            "You are a translator. \n\
            Please translate the following English text into the " + LANGUAGE + ":\n"
            + cur_sentence}]
    }

    r = None
    r_code = None
    is_success_sen = False
    is_translated_sen = False
    is_content_resp = False

    try_count_sen = 0
    re_translated_sen_count = 0

    translated_sen = ""

    while (not is_success_sen) or (not is_content_resp) or (not is_translated_sen):
        if try_count_sen == 3 or re_translated_sen_count == 2:
            print("ReTry Times Expired in Sentence Req! Discarded!")
            break

        try:
            is_success_sen = False
            is_translated_sen = False
            is_content_resp = False
            print("\n")
            # print(f"Current Try Count:{try_count_sen}")
            print("Request Sentence after 5s...")
            time.sleep(5)
            try_count_sen += 1
            print("###")
            r = post(url, headers=headers, json=sen_data_req, timeout=20)
            r_code = r.status_code
            print("#######")

            if r_code != 200:
                print(f"=== Sentence Req != 200 | Retry again! ===")
            else:
                is_success_sen = True

                ans = json.loads(r.content)
                content = ans['choices'][0]['message']['content']
                resp_translated_sen = content.strip()
                translated_sen = post_process_sentence(resp_translated_sen).replace('\xa0', ' ').strip()
                print("Translated Sentence:" + translated_sen)

                if len(translated_sen) != 0:
                    is_content_resp = True

                    if detect_language(translated_sen)[0].strip() == "en":
                        print("### ChatGPT did not translate Rel ! Retry again! ###")
                        re_translated_sen_count += 1
                    else:
                        is_translated_sen = True
                else:
                    print("=== Empty String Response! Retry again! ===")


        except:
            print(f"=== Sentence Req Exception | Retry again! ===")

    if (not is_success_sen) or (not is_content_resp) or (not is_translated_sen):
        continue

    print("\n")

############################# Triple Request ##########################
    trans_sub = []
    trans_rel = []
    trans_obj = []
    obejct_list = [''.join(obejct_list[i]) for i in range(len(obejct_list))]

    triple_iter = zip(subject_list, relation_list, obejct_list)

    for triple in triple_iter:
        triple_data_req = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": \
                "You are an Information Extraction expert. \n\
                The following are the extraction results of " + cur_sentence + ", which are represented by Subject, Relation, and Object:\
                \nSubject:" + triple[0] +
                "\nRelation:" + triple[1] +
                "\nObject:" + triple[2] +
                "\nPlease refer to the extraction results above, extracting a triple that corresponding Subject, Relation, and Object from translated sentence:\n"\
                + translated_sen +
                "\nNote that subject, relation and object must originate from the continuous segment of the sentence!\
                \nThe output format must be same with sample above."}]
        }

        response = None
        resp_code = None

        try_count_trip = 0
        re_translated_trip_count = 0
        content_list = []

        is_success_trip = False
        has_content_response = False
        is_translated_trip = False

        ret_sub = ""
        ret_rel = ""
        ret_obj = ""

        while (not is_success_trip) or (not has_content_response) or (not is_translated_trip):
            if try_count_trip == 4 or re_translated_trip_count == 2:
                print("ReTry Times Expired in Triple Req! Discarded!")
                break

            try:
                is_success_trip = False
                has_content_response = False
                is_translated_trip = False
                # print(f"Current Try Count:{try_count_trip}")
                print("Request Triple after 5s...")
                time.sleep(5)
                try_count_trip += 1

                response = post(url, headers=headers, json=triple_data_req, timeout=20)
                resp_code = response.status_code
                if resp_code != 200:
                    print(f"=== Triple Req != 200 | Retry again! ===")
                else:
                    # Request Succeed!
                    is_success_trip = True

                    answer = json.loads(response.content)
                    content = answer['choices'][0]['message']['content']  # translated sub rel obj
                    res_triple_list = content.split('\n')
                    # trimming empty string
                    content_list = [element for element in res_triple_list if element != ""]
                    print(f"Response Triple:{content_list}")

                    # Request Succeed and make sure Response 3 strings at least
                    if len(content_list) >= 3:
                        has_content_response = True

                        ret_sub_list = content_list[0].split(':')
                        ret_rel_list = content_list[1].split(':')
                        ret_obj_list = content_list[2].split(':')

                        ret_sub = post_process_triple(ret_sub_list).replace('\xa0', ' ').strip()
                        ret_rel = post_process_triple(ret_rel_list).replace('\xa0', ' ').strip()
                        ret_obj = post_process_triple(ret_obj_list).replace('\xa0', ' ').strip()

                        if ret_sub == "" or ret_rel == "" or ret_obj == "":
                            has_content_response = False
                            print("=== Empty Triple Element Response! Retry again! ===")
                        else:
                            # Response strings must be translated; return_relation cannot be original text
                            if ret_rel in cur_sentence:
                                if detect_language(ret_rel)[0].strip() == "en":
                                    print("### ChatGPT did not translate Rel ! Retry again! ###")
                                    re_translated_trip_count += 1
                                else:
                                    print(f"{LANGUAGE} words : ${ret_rel}$ Overlapped vocab words with English")
                                    is_translated_trip = True
                            else:
                                is_translated_trip = True

                    else:
                        print("=== Incomplete Triple Response! Retry again! ===")

            except:
                print("=== Triple Req Exception! Retry again! ===")

        if (not is_success_trip) or (not has_content_response) or (not is_translated_trip):
            is_Corrected_Triple_Generated = False
            break


############################## Check out  ###############################

        # check each element if it is matched with trans_sen
        # if it is not matched, try to replace each element by the most similar substring from original text
        if ret_sub not in translated_sen:
            print(f"Translated_sentence: {translated_sen}")
            print(f"Incorrect Sub: {ret_sub}")
            # try to revise incorrect element
            # result, start_index = try_correcting_triple(ret_sub, translated_sen)
            # if start_index != -1:
            #     # try to correct
            #     ret_sub = result
            #     print(f"Corrected Sub: {ret_sub}")
            # else:
            #     print("No similar substring in text! Sub cannot be corrected!")
            #     is_Matched = False
            #     break

            is_Matched = False
            break

        if ret_rel not in translated_sen:
            print(f"Translated_sentence: {translated_sen}")
            print(f"Incorrect Rel: {ret_rel}")
            # result, start_index = try_correcting_triple(ret_rel, translated_sen)
            # if start_index != -1:
            #     # try to correct
            #     ret_rel = result
            #     print(f"Corrected Rel: {ret_rel}")
            # else:
            #     print("No similar substring in text! Rel cannot be corrected!")
            #     is_Matched = False
            #     break
            is_Matched = False
            break

        if ret_obj not in translated_sen:
            print(f"Translated_sentence: {translated_sen}")
            print(f"Incorrect Obj: {ret_obj}")
            # result, start_index = try_correcting_triple(ret_obj, translated_sen)
            # if start_index != -1:
            #     # try to correct
            #     ret_obj = result
            #     print(f"Corrected Obj: {ret_obj}")
            # else:
            #     print("No similar substring in text! Obj cannot be corrected!")
            #     is_Matched = False
            #     break
            is_Matched = False
            break

        trans_sub.append(ret_sub)
        trans_rel.append(ret_rel)
        trans_obj.append(ret_obj)

        # print(f"Triple Saved: ret_sub:{ret_sub} ret_rel:{ret_rel} ret_obj:{ret_obj}")

#####################################################################################
    if not is_Corrected_Triple_Generated:
        print("\n### Cannot generate corrected Response Triple. Discarded ### ")
        continue

    if not is_Matched:
        print("\n### Triple cannot match with Sentence in this sample. Discarded ###")
        continue

    if len(trans_sub) == 0 or len(trans_rel) == 0 or len(trans_obj) == 0 or not (len(trans_sub) == len(trans_rel) == len(trans_obj)):
        print("\n### Bad data in response data. List length not Eq in triple. Discarded ###")
        continue

    print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ ALL Triples Matched! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print("Translated Subs:")
    print(trans_sub)
    print("Translated Rels:")
    print(trans_rel)
    print("Translated Objs:")
    print(trans_obj)

    valid_sample_data["sentence"] = translated_sen
    translated_sen_toks = tokenizer.tokenize(translated_sen)
    valid_sample_data["sentence_tokenized"] = translated_sen_toks
    # print(translated_sen_toks)

    check_repeated_triple_list = []
    tuple_item_iter = zip(trans_sub, trans_rel, trans_obj, confidence_score)

    is_malformed_pos = False
    for sub_item, rel_item, obj_item, score_item in tuple_item_iter:
        sub_item = sub_item.strip()
        rel_item = rel_item.strip()
        obj_item = obj_item.strip()

        print("\n")

        ### Delete Repeated Triple
        triple_list = [sub_item,rel_item,obj_item]
        if triple_list in check_repeated_triple_list:
            print("Repeated Triple!")
            continue
        else:
            check_repeated_triple_list.append(triple_list)

        # Final Triple saved
        print("#################################")
        # print(translated_sen)

        sub_item_toks = tokenizer.tokenize(sub_item)
        rel_item_toks = tokenizer.tokenize(rel_item)
        obj_item_toks = tokenizer.tokenize(obj_item)

        # print(sub_item_toks)
        # print(rel_item_toks)
        # print(obj_item_toks)

        sub_start = find_subarray_start(sub_item_toks, translated_sen_toks)
        rel_start = find_subarray_start(rel_item_toks, translated_sen_toks)
        obj_start = find_subarray_start(obj_item_toks, translated_sen_toks)


        #sub_start = translated_sen.index(sub_item)
        sub_end = sub_start + len(sub_item_toks) - 1
        #rel_start = translated_sen.index(rel_item)
        rel_end = rel_start + len(rel_item_toks) - 1
        #obj_start = translated_sen.index(obj_item)
        obj_end = obj_start + len(obj_item_toks) - 1

        if sub_start == -1 or sub_end == -1 or rel_start == -1 or rel_end == -1 or obj_start == -1 or obj_end == -1:
            is_malformed_pos = True
            break

        print(f"sub_start:{sub_start} sub_end:{sub_end}")
        print(f"rel_start:{rel_start} rel_end:{rel_end}")
        print(f"obj_start:{obj_start} obj_end:{obj_end}")



        tuple_ele = {
                "score": score_item,
                "context": "None",
                "arg0": sub_item,
                "relation": rel_item,
                "args": [
                    obj_item
                ],
                "arg0_pos": [
                    sub_start,
                    sub_end
                ],
                "rel_pos": [
                    rel_start,
                    rel_end
                ],
                "args_pos": [
                    [
                        obj_start,
                        obj_end
                    ]
                ]
        }

        valid_sample_data["tuples"].append(tuple_ele)
        # print(valid_sample_data)
    if is_malformed_pos:
        continue
    
    try:
        #json.dump(valid_sample_data, new_json_file) # multi languages
        json.dump(valid_sample_data, new_json_file, ensure_ascii=False) # only Chinese
        new_json_file.write(os.linesep)
        new_json_file.flush()

    except:
        print("\n### File Writing Error ###")
        continue

    print("\nCongrats! File Writing Successful!")
    ALL_MATCHED_SAMPLE_NUM += 1

new_json_file.close()
print("TOTAL TIME: ", time.time() - start)
print(f"Finished! All Valid Sample Number: {ALL_MATCHED_SAMPLE_NUM}")