# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FirstscraperItem(scrapy.Item):
    title = scrapy.Field() # 文本字段
    #publish_date = scrapy.Field() # 日期字段
    #url = scrapy.Field()
    #url_object_id = scrapy.Field()
    #content = scrapy.Field()
    purecontent = scrapy.Field()
    
    
    
    
    
