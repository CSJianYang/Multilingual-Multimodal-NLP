import os 
import threading
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

useless_set = ['javascript:void(0);', 'mailto:admin@runoob.com', 'javascript:;']

visited_urls = set()

def file_in(file, urls):
    if urls == []:
        return 
    with open(file, 'a') as f:
        for link in urls:
            if link not in useless_set:
                f.write(link + '\n')
        f.close()

def get_links(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        links = []
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            link = urljoin(url, link)
            links.append(link)
        return links
    except requests.exceptions.RequestException as e:
        print(f'Error fetching {url}: {e}')
        return []
    

def dfs_crawl(url, depth):
    if depth == 0 or url in visited_urls:
        return
    
    print(f"Crawling {url} at depth {depth}")

    visited_urls.add(url)
    links = get_links(url)
    file_in('indexes.txt',links)

    for link in links:
        t = threading.Thread(target=dfs_crawl, args=(link, depth-1))
        t.start()


if __name__ == '__main__':
    start_url = "https://www.runoob.com"
    max_depth = 2

    t = threading.Thread(target=dfs_crawl, args=(start_url, max_depth))
    t.start()
    t.join()