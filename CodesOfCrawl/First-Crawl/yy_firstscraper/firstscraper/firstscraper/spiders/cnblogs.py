import scrapy
import urllib.parse
# from firstscraper import items   #
from datetime import datetime
from scrapy.selector import Selector
from firstscraper.items import FirstscraperItem


class CnblogsSpider(scrapy.Spider):
    name = "cnblogs"    
    allowed_domains = ["www.cnblogs.com"]  
    start_urls = ["https://www.cnblogs.com/pick/"]  

    def parse(self, response):     
        post_nodes = response.css('#post_list .post-item-text')
        for post_node in post_nodes:
            post_url = post_node.css('a::attr(href)').extract_first('')
            yield scrapy.Request(url=urllib.parse.urljoin(response.url,post_url),callback=self.parse_detail)
        # 以上：获取精选列表页的url交给scrapy进行下载后调用相应的解析方法
        
        # next_url = response.xpath('//div[@class="pager"]/a[contains(text(), ">")]/@href').extract_first('')
        next_url = response.css('div.pager a:last-child::text').extract_first('')  # 因为最后的界面没有'>'
        if next_url == '>':
            next_url = response.css('div.pager a:last-child::attr(href)').extract_first('')
            yield scrapy.Request(url=urllib.parse.urljoin(response.url,next_url),callback=self.parse)
        # 以上：获取下一页的url，下载，解析
        
    def parse_detail(self, response):
        #title = response.css('#topics span::text').extract_first('')
        title_element = response.xpath('//span[@role="heading"][@aria-level="2"]')
        title = title_element.xpath('text()').extract_first('')
        publish_date = response.css('#topics .postDesc span::text').extract_first('')
        content_elements = response.css('#cnblogs_post_body')
        if content_elements:
            content = response.css('#cnblogs_post_body').extract()[0] 
        else:
            content = ""
       

        content_paragraphs = response.xpath('//div[@id="cnblogs_post_body"]//p')
        pure_content = '\n'.join(content_paragraphs.xpath('string()').extract())
     
        
        article_item = FirstscraperItem()
        article_item['title'] = title
        #article_item['publish_date'] = publish_date
        #article_item['content'] = content
        article_item['purecontent'] = pure_content
        #article_item['url'] = response.url
            
        yield article_item
            
