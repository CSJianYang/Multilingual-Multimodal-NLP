import requests
from pyquery import PyQuery as pq
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin


url = 'https://www.runoob.com/'
html = requests.get(url).decode()

with open('htmls.html', 'w+') as t:
    t.write(html)
t.close()

soup = BeautifulSoup(html, 'html.parser')

links = soup.find_all('a')



with open('indexes.txt','w+') as f:
    for link in links:
        href = link.get('href')
        if href:
            if not (href.startswith('https://') or href.startswith('http://')):
                href = urljoin(url,href)
            f.write(href + '\n')
f.close()