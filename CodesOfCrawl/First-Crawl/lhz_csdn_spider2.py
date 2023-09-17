import os
from html import unescape

import requests
from html2text import HTML2Text
from lxml import etree


def crawl(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.69",
    }
    print("crawl...")

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        html = response.content.decode("utf8")
        # print(html)
        tree = etree.HTML(html)
        # 找到需要的html块
        title = tree.xpath('//*[@id="articleContentId"]/text()')[0]
        block = tree.xpath('//*[@id="content_views"]')
        # html
        ohtml = unescape(etree.tostring(block[0]).decode("utf8"))
        # 纯文本
        text = block[0].xpath('string(.)').strip()
        print("title:", title)
        save(ohtml, text, title)
        print("finish!")
    else:
        print("failed!")


def save(html, text, title):
    if "output" not in os.listdir():
        os.mkdir("output")
        os.mkdir("output/html")
        os.mkdir("output/text")
        os.mkdir("output/markdown")
    with open(f"output/html/{title}.html", 'w', encoding='utf8') as html_file:
        # 保存html
        print("write html...")
        html_file.write(html)
    with open(f"output/text/{title}.txt", 'w', encoding='utf8') as txt_file:
        # 保存纯文本
        print("write text...")
        txt_file.write(text)
    with open(f"output/markdown/{title}.md", 'w', encoding='utf8') as md_file:
        # 保存markdown
        print("write markdown...")
        text_maker = HTML2Text()
        # md转换
        md_text = text_maker.handle(html)
        md_file.write(md_text)


if __name__ == '__main__':
    # url = "https://blog.csdn.net/seanyang_/article/details/126571472?spm=1001.2101.3001.6650.4&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-126571472-blog-99157971.235%5Ev38%5Epc_relevant_sort_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-4-126571472-blog-99157971.235%5Ev38%5Epc_relevant_sort_base2&utm_relevant_index=9"
    url_list = []
    for url in url_list:
        crawl(url)
