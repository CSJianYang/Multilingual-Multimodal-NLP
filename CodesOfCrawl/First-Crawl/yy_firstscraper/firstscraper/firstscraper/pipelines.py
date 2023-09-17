import codecs
import json
import os  # 导入 os 模块

from itemadapter import ItemAdapter

class JsonWithPipeline(object):
    def __init__(self):
        # 使用 os.path.join 来构建文件路径
        self.file = codecs.open(os.path.join('output', 'purearticle.json'), 'a', encoding='utf-8')

    def process_item(self, item, spider):
        lines = json.dumps(dict(item), ensure_ascii=False) + '\n'
        self.file.write(lines)
        return item

    def spider_closed(self, spider):
        self.file.close()
