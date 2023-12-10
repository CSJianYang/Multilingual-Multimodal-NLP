from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

# 加载大模型板块
model_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()


# 加载服务器板块
app = Flask(__name__)

# 输出预处理板块
def post_process(dic):
    # dic: [[op, pos], [op, pos], ...]
    new_dic = []
    for i in range(len(dic)):
        new_dic.append(dic[i])
    new_dic.sort(key=lambda x: (x[0], x[1]))
    # new_dic: [[op, pos], [op, pos], ...] 排序后并且删除了一些不满足条件的[op, pos]
    # print(dic)
    result = ""
    for i in new_dic:
        result += (hex(i[0])[2:].rjust(3, '0') + hex(i[1])[2:].rjust(3, '0'))
    # print(result)
    result += "00e001"
    # result: 16进制每3位表示一个数字
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", "-p", type=str, default="objdump", help="objdump, readelf, tiffsplit, nm, mp3gain, magick, libxml, libjpeg")
    args = parser.parse_args()
    return args


@app.route('/')
def get_output():
    args = parse_args()
    # 这个函数是主要用于输入得到的input，input是16进制的字符串，如"0011223344..."，但保证长度为2的倍数
    data = request.args.get('input', '')
    # print(data)
    data = data.strip().replace('\n', '').replace('\r', '')
    if data == '':
        return []
    data = [data[i * 2: i * 2 + 2] for i in range(len(data) // 2)]
    byte_input = ['0x' + hex(int(i, 16))[2:].rjust(2, '0') for i in data]
    # byte_input: "0x"
    dataset_name = args.program
    input_text = f'''Task description:Now, you are a AFL (American Fuzzy Lop), which is a highly efficient and widely used fuzz testing tool designed for finding security vulnerabilities and bugs in software. You are now fuzzing a program named {dataset_name}, which requires variable (a byte sequence) to run. I will give you a byte sequence as input sequence, and you need to mutate the input sequence to give me a output sequence through a mutation operation below. Finally you need to give me a output which includes input sequence, mutation operation and output sequence.
Mutation operations:1. Perform bitfilp on a bit randomly.2. Perform bitfilp on two neighboring bits randomly.3. Perform bitfilp on four neighboring bits randomly.4. Randomly select a byte and XOR it with 0xff.5. Randomly select two neighboring bytes and XOR them with 0xff.6. Randomly select four neighboring bytes and XOR them with 0xff.7. Randomly select a byte and perform addition or subtraction on it (operands are 0x01~0x23).8. Randomly select two neighboring bytes and convert these two bytes into a decimal number. Select whether to swap the positions of these two bytes. Perform addition or subtraction on it (operands are 1~35). Finally convert this number to 2 bytes and put it back to its original position.9. Randomly select four neighboring bytes. Select whether to swap the positions of these four bytes. Convert these four bytes into a decimal number. Perform addition or subtraction on it (operands are 1~35). Finally convert this number to 4 bytes and put it back to its original position.10. Randomly select a byte and replace it with a random byte in {{0x80, 0xff,0x00,0x01,0x10,0x20,0x40,0x64,0x7F}}.11. Randomly select two neighboring bytes and replace them with two random bytes in {{(0xff 0x80),(0xff 0xff),(0x00 0x00),(0x00 0x01),(0x00 0x10),(0x00 0x20),(0x00 0x40),(0x00 0x64),(0x00 0x7f),(0x80 0x00),(0xff 0x7f),(0x00 0x80),(0x00 0xff),(0x01 0x00),(0x02 0x00),(0x03 0xe8),(0x04 0x00),(0x10 0x00),(0x7f 0xff)}}.12. Randomly select four neighboring bytes and replace them with four random bytes in {{(0xff 0xff 0xff 0x80),(0xff 0xff 0xff 0xff),(0x00 0x00 0x00 0x00),(0x00 0x00 0x00 0x01),(0x00 0x00 0x00 0x10),(0x00 0x00 0x00 0x20),(0x00 0x00 0x00 0x40),(0x00 0x00 0x00 0x64),(0x00 0x00 0x00 0x7f),(0xff 0xff 0x80 0x00),(0xff 0xff 0xff 0x7f),(0x00 0x00 0x00 0x80),(0x00 0x00 0x00 0xff),(0x00 0x00 0x01 0x00),(0x00 0x00 0x02 0x00),(0x00 0x00 0x03 0xe8),(0x00 0x00 0x04 0x00),(0x00 0x00 0x10 0x00),(0x00 0x00 0x7f 0xff),(0x80 0x00 0x00 0x00),(0xfa 0x00 0x00 0xfa),(0xff 0xff 0x7f 0xff),(0x00 0x00 0x80 0x00),(0x00 0x00 0xff 0xff),(0x00 0x01 0x00 0x00),(0x05 0xff 0xff 0x05),(0x7f 0xff 0xff 0xff)}}.
Input Sequence Definition:It consists of bytes represented in hexadecimal, separated by spaces. It is the byte sequence to be mutated. It is a variable that can cause the program to crash or trigger a new path.
Output Sequence Definition:It consists of bytes represented in hexadecimal, separated by spaces. It is the mutated byte sequence. It is a variable that can cause the program to crash or trigger a new path.
input sequence:{byte_input}Please list all possible mutation strategies (mutation position and mutation operation) with the JSON format as:output:{{    "mutation strategies": [        (op_1, pos_1),         (op_2, pos_2),         ... ,         (op_N, pos_N)    ]}}'''
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    # 这里应该是得到4个输出，所以下面用了for循环
    outputs = model.generate(**inputs, max_length=128, topp=0.5, sampling_num=4)
    output_results = tokenizer.decode(outputs[0], skip_special_tokens=True)
    dic = []
    for output_one in output_results:
        for i in output_one["mutation strategies"]:
            if i not in dic:
                dic.append(i)
    # dic: [[op, pos], [op, pos], ...] (list)
    # 必须在发送result之前用post_process处理dic再发送给client
    result = post_process(dic)
    return result

if __name__ == '__main__':
    # 如果5000端口被占用了，注意换下面的port，并保持module_client.py里面的port是一致的
    app.run(port=5000,debug = True)
