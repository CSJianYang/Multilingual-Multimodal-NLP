import os,requests,lxml
from bs4 import BeautifulSoup


for time in range(260):
    with (open(f'./index/{time}/tmp.html', 'r+') as f1):
        content = f1.read()
        soup = BeautifulSoup(content, 'lxml')
        p_list = soup.find_all('p')
        with open(f'./index/{time}/{time}.txt','a') as f2:
            f2.write('\n'.join(str(p) for p in p_list))
            f2.close()
        f1.close()
