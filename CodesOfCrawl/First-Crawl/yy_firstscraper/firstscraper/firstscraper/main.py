import sys
import os
from scrapy.cmdline import execute

#将当前文件添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#脚本方式一键运行scrapy项目
execute(['scrapy','crawl','cnblogs'])
