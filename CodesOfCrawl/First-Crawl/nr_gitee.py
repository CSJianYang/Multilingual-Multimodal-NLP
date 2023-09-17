# # 爬取gitee内容
# # file_name1.txt - file_content1.txt
# # file_name2.txt - file_content2.txt
# # ...
# import os
# import re
#
import json
import os
import sys

from lxml import etree
import requests
# my_url = 'https://gitee.com/cheng-zhipeng-1/flyd?_from=gitee_search'
#
#
# my_header = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0'
# }
#

# 1. 确定一个用户名
# 2. 拉取所有仓库名
# 3. 遍历所有仓库
# 4. 对每个仓库的master分支, 递归遍历所有文件


def main():
    # get_user_all_files('cheng-zhipeng-1')
    ret_names = get_user_repo_contribution('dromara', 'hutool')
    for user_name in ret_names:
        get_user_all_files(user_name)


def get_user_all_files(username: str):
    user_projects_url = f'https://gitee.com/{username}/projects'
    my_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0'
    }
    response = requests.get(url=user_projects_url, headers=my_header)
    et = etree.HTML(response.content)
    # path = tianyalei/ParSeq
    warehouse_paths = et.xpath('//*[@class="project list-warpper"]/@data-path')
    # warehouse_paths = et.xpath('//*[@id="tree-slider"]/@class')


    for warehouse_path in warehouse_paths:
        # 遍历所有仓库路径
        warehouse_url = f'https://gitee.com/{warehouse_path}'
        get_warehouse_all_files(warehouse_url, '')  # 拿到某个用户名的仓库的所有文件


# https://gitee.com/cheng-zhipeng-1/flyd
def get_warehouse_all_files(warehouse_url: str, cur_path: str):
    # 拿到仓库的所有路径
    my_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0'
    }
    pre = '/'
    if cur_path != '':
        pre += 'tree/master/'
    this_warehouse_url = warehouse_url + pre + cur_path
    response = requests.get(url=this_warehouse_url, headers=my_header)
    et = etree.HTML(response.text)  # 仓库页下的html, 只爬master  不能用response.content, 不含编码信息
    master_files = et.xpath('//*[@id="tree-slider"]//div[@data-branch="master"]')
    for e in master_files:
        file_type = e.xpath('div[1]/@data-type')[0]
        file_name = e.xpath('div[1]/@data-path')[0]  # .encode('latin-1').decode('utf-8')
        if file_type == 'folder':  # 目录
            # os.mkdir('result/' + file_name)
            get_warehouse_all_files(warehouse_url, file_name)
        elif check_filename_is_download(str(file_name)):
            # 下载这个文件 同名的话跳过
            download_file_url = warehouse_url + '/raw/master/' + file_name
            response = requests.get(url=download_file_url, headers=my_header)
            with open('result/' + get_real_file_name_from_path(file_name), 'wb') as file:
                file.write(response.content)


def dir_is_contain_filename(dir_path, file_name):
    files = os.listdir(dir_path)
    if file_name in files:
        return True
    return False


def check_filename_is_download(file_name: str):
    goal_types = ['.c', '.java', '.py', '.cpp', '.h', '.hpp', '.js', '.rb', '.php', '.swift', '.go', '.rs', '.kt',
                  '.html', '.htm', '.sh', '.bash', '.sql', '.pl', '.m', '.r', '.json', '.txt', 'readme.md', 'README.md']
    for goal_type in goal_types:
        if file_name.endswith(goal_type):
            return True
    return False


def get_real_file_name_from_path(file_path: str):
    index = file_path.find('/')
    if index != -1:
        # 找到最后一个斜杠
        last_index = 0
        for i, c in enumerate(file_path):
            if c == '/':
                last_index = i
        return file_path[last_index + 1:]
    else:
        return file_path


def get_master_dir_all_files(last_url, dir_name):
    new_url = ''


def get_user_repo_contribution(user_name: str, repo_name: str):
    get_url = f'https://gitee.com/api/v5/repos/{user_name}/{repo_name}/contributors'
    get_header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0'
    }
    response = requests.get(url=get_url, headers=get_header)
    # print(response)
    # print(response.content.decode('utf-8'))
    json_data = json.loads(response.content.decode('utf-8'))
    ret_names = []
    for e in json_data:
        ret_names.append(e['name'])
    return ret_names
    # print(response.text)
    # print(json_data)

# import requests
#
# # 文件的原始 URL
# # file_url = 'https://gitee.com/用户名/仓库名/raw/分支/文件路径/文件名'
# file_url = 'https://gitee.com/cheng-zhipeng-1/flyd/raw/master/测试数据.docx'
# # 发送 GET 请求以获取文件内容
# response = requests.get(file_url)
#
# # 检查响应状态码，确保请求成功
# if response.status_code == 200:
#     # 提取文件名，你可以从文件URL中提取或手动指定
#     file_name = '3.docx'  # 替换为你想要的文件名和扩展名
#     # 以二进制写入方式打开文件，并保存响应内容
#     with open(file_name, 'wb') as file:
#         file.write(response.content)
#     print(f'文件已下载为: {file_name}')
# else:
#     print('下载失败，状态码:', response.status_code)

# import requests
# import os
#
# def download_user_files(username: str):
#     # 构建用户主存储库的URL
#     user_repository_url = f'https://gitee.com/{username}/czp'
#
#     # 发送 GET 请求以获取存储库内容
#     response = requests.get(user_repository_url)
#
#     # 检查响应状态码，确保请求成功
#     if response.status_code == 200:
#         # 创建存储用户文件的目录
#         if not os.path.exists(username):
#             os.mkdir(username)
#
#         # 提取存储库的HTML内容
#         html_content = response.text
#         # 在HTML内容中查找文件链接，并下载每个文件
#         start_index = 0
#         while True:
#             # 查找文件链接的起始位置
#             file_link_start = html_content.find('<a class="codelink"', start_index)
#
#             # 如果没有找到更多文件链接，退出循环
#             if file_link_start == -1:
#                 break
#             # 查找文件链接的结束位置
#             file_link_end = html_content.find('">', file_link_start)
#
#             # 提取文件名和URL
#             file_link = html_content[file_link_start:file_link_end]
#             file_name_start = file_link.find('>') + 1
#             file_name = file_link[file_name_start:]
#             file_url_start = file_link.find('href="') + 6
#             file_url = file_link[file_url_start:]
#
#             # 下载文件
#             file_response = requests.get(file_url)
#             if file_response.status_code == 200:
#                 with open(os.path.join(username, file_name), 'wb') as file:
#                     file.write(file_response.content)
#                     print(f'Downloaded: {file_name}')
#             else:
#                 print(f'Failed to download: {file_name}')
#
#             # 更新查找的起始位置
#             start_index = file_link_end
#     else:
#         print('Failed to fetch user repository content.')
#
# download_user_files('cheng-zhipeng-1')

if __name__ == '__main__':
    main()