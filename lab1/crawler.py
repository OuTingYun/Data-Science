import sys
import urllib
import json
import time

import requests
from bs4 import BeautifulSoup as BS
import re
from datetime import datetime
# import jsonlines

ptt = "https://www.ptt.cc"


def fun():
    args = sys.argv[1:]
    fun_type = 0
    if len(args) == 1 and args[0] == 'crawl':
        fun_type = 1
    elif len(args) == 3 and args[0] == 'push':
        fun_type = 2
    elif len(args) == 3 and args[0] == 'popular':
        fun_type = 3
    elif len(args) == 4 and args[0] == 'keyword':
        fun_type = 4
    return args, fun_type


def ptt_crawl():
    start_time = time.time()
    # find starting page
    start = 3643
    end = 3955

    for index in range(start, 0, -1):
        time.sleep(0.5)
        url = f'https://www.ptt.cc/bbs/Beauty/index{index}.html'
        # 一直向 server 回答滿 18 歲了 !
        response = requests.get(url, cookies={'over18': '1'})
        content = response.text
        print("finding start page:", index, end='\r')
        if re.findall("羅志祥強勢回歸", content) != []:
            start = index
            print(" "*25, end='\r')
            break

    for index in range(start+305, start+350, 1):
        time.sleep(0.5)
        url = f'https://www.ptt.cc/bbs/Beauty/index{index}.html'
        # 一直向 server 回答滿 18 歲了 !
        response = requests.get(url, cookies={'over18': '1'})
        content = response.text
        print("finding end page:", index, end='\r')
        if re.findall("孟潔MJ", content) != []:
            end = index
            print(" "*20, end='\r')
            break

    print("start:", start, " end:", end)
    finding_page = []
    for index in range(start, end+1):
        time.sleep(0.5)
        url = f'https://www.ptt.cc/bbs/Beauty/index{index}.html'
        # 一直向 server 回答滿 18 歲了 !
        response = requests.get(url, cookies={'over18': '1'})
        content = response.text
        soup = BS(content, 'html.parser')
        finding_page.append(soup)
        print("crawling=>", index, end="\r")

    articles = []
    for idx, soup in enumerate(finding_page):
        article = soup.find_all(class_='r-ent')
        if idx == 0:
            article = article[2:]
        elif idx == len(finding_page)-1:

            article = article[:-4]
        articles += article

    print("spent time:", time.time()-start_time)
    return articles


def crawl(articles):

    start_time = time.time()
    filter_title = '(\[公告\])'

    articles_info = []
    explores_info = []
    gotGUY = False
    gotGirl = False
    for article in articles:
        title = article.find(class_='title').text.replace("\n", "")
        if title != "[帥哥] 羅志祥強勢回歸" and not gotGUY:
            continue
        elif title == "[帥哥] 羅志祥強勢回歸":
            gotGUY = True

        if gotGirl:
            continue
        elif title == "[正妹] 孟潔MJ ":
            gotGirl = True

        ann = re.findall(filter_title, title)
        if ann != []:
            continue
        if article.find(class_='title').a == None:
            print("no article.find(class_='title').a")
            continue
        if 'href' not in article.find(class_='title').a.attrs.keys():
            print("No url ", title)
            continue
        url = ptt + article.find(class_='title').a['href']

        dates = article.find(class_='date').text.split('/')
        date = datetime(year=2022, month=int(
            dates[0]), day=int(dates[1])).strftime("%m%d")

        info = {"date": date, "title": title, "url": url}
        articles_info.append(info)

        if article.find(class_='nrec').text == '爆':
            explores_info.append(info)
    print("spent time:", time.time()-start_time)
    return articles_info, explores_info


def push(articles_info, args):
    start_time = time.time()
    like_person = {}
    boo_person = {}
    like_count = 0
    boo_count = 0

    start_date = args[1]
    end_date = args[2]

    for idx, page in enumerate(articles_info):
        if page['date'] < start_date or page['date'] > end_date:
            continue
        print("page==>", idx, "/", len(articles_info), end='\r')
        time.sleep(0.1)
        content = requests.get(page['url'], cookies={'over18': '1'}).text
        soup = BS(content, 'html.parser')
        pushes = soup.find_all(class_='push')
        for push in pushes:
            span = push.find_all('span')
            reaction = span[0].text[0]

            name = span[1].text
            if reaction == '推':
                if name not in like_person:
                    like_person[name] = 0
                like_person[name] += 1
                like_count += 1
            elif reaction == '噓':
                if name not in boo_person:
                    boo_person[name] = 0
                boo_person[name] += 1
                boo_count += 1

    statistics = {"all-like": like_count, "all-boo": boo_count}
    store = [{f"like {idx+1}": {"user_id": k, "count": v}} for idx, (k, v) in enumerate(dict(
        sorted(like_person.items(), key=lambda item: (item[1], -ord(item[0][0])), reverse=True)).items())]
    for idx, ll in enumerate(store):
        item = list(ll.values())[0]
        if idx < 10:
            statistics[f"like {idx+1}"] = {"user_id": item["user_id"],
                                           "count": item["count"]}
        else:
            break

    store = [{f"like {idx+1}": {"user_id": k, "count": v}} for idx, (k, v) in enumerate(dict(
        sorted(boo_person.items(), key=lambda item: (item[1], -ord(item[0][0])), reverse=True)).items())]
    for idx, ll in enumerate(store):
        item = list(ll.values())[0]
        if idx < 10:
            statistics[f"boo {idx+1}"] = {"user_id": item["user_id"],
                                          "count": item["count"]}
        else:
            break

    print("spent time:", time.time()-start_time)
    export(f"push_{start_date}_{end_date}.json", [statistics])


def popular(explores_info, args):

    start_time = time.time()
    start_date = args[1]
    end_date = args[2]

    explore_count = 0

    http_filter = 'https?://.*[png|jpg|jpeg|gif|PNG|JPG|JPEG|GIF]'
    http_list = []
    for idx, page in enumerate(explores_info):
        if page['date'] < start_date or page['date'] > end_date:
            continue
        print("date==>", page['date'], " idx===>", idx, end='\r')
        explore_count += 1

        time.sleep(0.5)
        content = requests.get(page['url'], cookies={'over18': '1'}).text
        soup = BS(content, 'html.parser')
        http = re.findall(http_filter, soup.text)
        http_list += [url for url in http if re.findall(
            "(png)|(jpg)|(jpeg)|(gif)|(PNG)|(JPG)|(JPEG)|(GIF)", url) != []]

    populars = {"number_of_popular_articles": explore_count}
    populars["image_urls"] = [url for url in http_list]

    print("spent time:", time.time()-start_time)
    export(f"popular_{start_date}_{end_date}.json", [populars])
    # export(f"articles_{start_date}_{end_date}.json",[populars])


def keyword(articles_info, args):
    start_time = time.time()
    keyword_filter = args[1]
    start_date = args[2]
    end_date = args[3]

    http_filter = 'https?://.*[png|jpg|jpeg|gif|PNG|JPG|JPEG|GIF]'
    http_list = []

    for idx, page in enumerate(articles_info):
        if page['date'] < start_date or page['date'] > end_date:
            continue
        print("date==>", page['date'], " idx===>", idx, end='\r')
        time.sleep(0.5)
        content = requests.get(page['url'], cookies={'over18': '1'}).text
        soup = BS(content, 'html.parser')

        soups = soup.text.split('※ 發信站')
        if len(soups) != 2:
            continue
        keyword_content = soups[0].split('作者')[1]
        if len(re.findall(keyword_filter, keyword_content)) == 0:
            continue

        http = re.findall(http_filter, soup.text)
        http_list += [url for url in http if re.findall(
            "(png)|(jpg)|(jpeg)|(gif)|(PNG)|(JPG)|(JPEG)|(GIF)", url) != []]
    keywords = {"image_urls": [url for url in http_list]}

    print("spent time:", time.time()-start_time)
    export(
        f"keyword_{keyword_filter}_{start_date}_{end_date}.json", [keywords])


def export(fname, l):
    with open(fname, 'w', encoding='utf8') as f:
        for item in l:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def load_articles():
    articles_info = []
    explores_info = []
    # with jsonlines.open('all_article.jsonl') as reader:
    #     for row in reader:
    #         articles_info.append(row)
    # with jsonlines.open('all_popular.jsonl') as reader:
    #     for row in reader:
    #         explores_info.append(row)

    # data = []
    with open('all_article.jsonl', encoding="utf-8") as reader:
        for row in reader:
            articles_info.append(json.loads(row))
    with open('all_popular.jsonl', encoding="utf-8") as reader:
        for row in reader:
            explores_info.append(json.loads(row))
    print(len(articles_info))

    return articles_info, explores_info


if __name__ == '__main__':
    args, fun_type = fun()

    if fun_type == 1:
        articles = ptt_crawl()
        articles_info, explores_info = crawl(articles)
        export("all_article.jsonl", articles_info)
        export("all_popular.jsonl", explores_info)
    else:
        articles_info, explores_info = load_articles()
        if fun_type == 2:
            push(articles_info, args)
        elif fun_type == 3:
            popular(explores_info, args)
            # popular(articles_info,args)
        elif fun_type == 4:
            keyword(articles_info, args)
        else:
            assert ("Error type")
