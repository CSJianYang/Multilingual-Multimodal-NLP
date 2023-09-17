#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   content_get.py
@Time    :   2023/09/09 13:57:01
@Author  :   Yan Zhimin 
@Version :   1.0
@Contact :   a13684703281@163.com
@License :   None
@Desc    :   None
'''

'''
# 菜鸟教程爬虫html观察
0. base_url = "https://www.runoob.com/html/html-links.html"
1. 正文类 class="article-intro"
2. 正文<p> 换行<br> 列表<ul> 代码段<pre class="prettyprint prettyprinted" style>
3. 斜体<i> 粗体<strong>
代码段用```来分割
将文本内容以md格式进行映射
按文本框架来进行整理数据
{
    "": "",
    "介绍": "",
    "段落": "",
    "列表": "",
    "代码段": "",
}
ip池挖掘
控制时间间隔
多进程的实现

'''

# here put the import lib

from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import requests
import threading
import os



file0 = open('indexes.txt', 'r')
file1 = open('contents.txt', 'a')
urls = file0.readlines()
visited_urls = set()


def file_in(file, url, article, para, ul, code):
    file.write(f'The messages about {url} is:\n' + '-' * 20 + '\n')
    file.write('正文类所有内容如下：\n')
    if article is not None:
        file.write(article)
    file.write('正文段落信息如下：\n')
    if para is not None:
        for p in para:
            file.write(p + '\n')
    file.write('列表信息如下：\n')
    if ul is not None:
        for u in ul:
            file.write(u + '\n')
    file.write('代码段信息如下：\n')
    if code is not None:
        for c in code:
            file.write(c + '\n')
    file.write('-'*20+'\n')


def get_contents(url,time):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_intro, paragraphs, ul_lists, code_snippets = [], [], [], []
        article_intro_elements = soup.find('div', class_='article-intro')
        if article_intro_elements:
            article_intro = article_intro_elements.text.strip()
        else:
            article_intro = None
        paragraph_elements = soup.find_all('p')
        if paragraph_elements:
            paragraphs = [p.text.strip() for p in paragraph_elements]
        else:
            paragraphs = []
        ul_lists_elements = soup.find_all('ul')
        if ul_lists_elements:
            ul_lists = [ul.text.strip() for ul in ul_lists_elements]
        else:
            ul_lists = []
        code_snippets_elements = soup.find_all('pre', class_='prettyprint prettyprinted')    
        if code_snippets_elements:
            code_snippets = [pre.text.strip() for pre in code_snippets_elements]
        else:
            case_list = []
        with open(f'{time}.txt', 'a') as file:
            file_in(file=file, url=url, article=article_intro,
                para=paragraphs, ul=ul_lists, code=code_snippets)
    except requests.exceptions.RequestException as e:
        print(f'Error fetching {url}: {e}')
        return 


def crawl():
    time = 0
    for url in urls:
        if url in visited_urls:
            continue
        else:
            visited_urls.add(url)
            get_contents(url=url,time=time)
            time +=1

if __name__ == '__main__':
    crawl()
    file0.close()
    file1.close()