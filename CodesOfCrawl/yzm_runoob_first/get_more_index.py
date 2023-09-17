import os
import requests
import lxml
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = 'https://www.runoob.com/'

with open('indexes.txt', 'r+') as f1:
    time = 0
    for url in f1.readlines():
        if (url is not None) and (url.startswith('https://') or url.startswith('http://')):  # 这里是进行挑选有用的url
            html = ''
            with open(f'./index/{time}/tmp.html', 'w+', encoding='utf-8') as tmp:
                tmp.write(requests.get(url).text)
                tmp.seek(0)
                html = tmp.read()
            if not os.path.exists(f'./index/{time}'):
                os.mkdir(f'./index/{time}')
            with open(f'./index/{time}/{time}.html', 'a') as f2:
                f2.write(html) 
                
            f2.close()
            soup = BeautifulSoup(html, 'lxml')
            links = soup.find_all('a')
            
            for link in links:
                href = link.get('href')
                if (href is not None) and (href.startswith('https://') or href.startswith('http://')):
                    with open(f'./index/{time}/indexes.txt','a') as f3:
                        href = urljoin(url,href)
                        f3.write(href + '\n')
                    f3.close()
            time += 1