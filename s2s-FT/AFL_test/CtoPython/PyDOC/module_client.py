import requests
import torch
import os
pad = 499

def writefragment(mes):
    with open("../docset/fragment.txt", "a") as f:
        f.write(repr(mes))
        f.write("\n")

def request_data(data):
    url = 'http://127.0.0.1:80/'
    # 这里需要根据module_app里面的设置动态调整，现在调整的是一样的
    params = {'input': data}
    response = requests.get(url, params=params)
    return response.text


def run(data):
    data = data.strip().replace('\n', '').replace('\r', '')
    result = request_data(data)
    return result
    
def readdoc():
    with open("../docset/in.txt", 'r') as f:
        content = f.read()
    return content
    
def writedoc(data):
    with open("../docset/out.txt", "w") as f:
    	f.write(data)
    	
def writeerr(mes):
    with open("../docset/err.txt", "w") as f:
    	f.write(repr(mes))
    	
if __name__ == '__main__':
    data = readdoc()
    try:
        result = run(data)
    except Exception as e:
        writeerr(e)
        result=""
    writedoc(result)
        
