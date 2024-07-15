import os.path
import re
import time
from datetime import datetime, timedelta
import pymysql
import wget
import requests
import gzip
import io
from glob import glob
import json
from tqdm import tqdm
from urllib.parse import urlparse


def download_and_extract_json_gz(url, output_json_filename):
    # 发送GET请求获取文件
    response = requests.get(url, stream=True)

    # 检查响应状态码是否为200
    if response.status_code == 200:
        # 使用BytesIO和gzip解压文件
        with io.BytesIO(response.content) as f_in:
            with gzip.GzipFile(fileobj=f_in, mode='rb') as f_gz:
                # 读取解压后的内容
                json_data = str(f_gz.read())[2:]

                with open(output_json_filename, 'w', encoding='utf-8') as f:
                    f.write(json_data)

        print(f"JSON文件已成功下载并解压到{output_json_filename}")
    else:
        print(f"下载失败，状态码：{response.status_code}")


def get_github_menu():
    # 设置起始日期为2020年1月1日
    start_date = datetime(2024, 6, 3)

    # 获取当前日期
    current_date = datetime.now()

    dir = r"G:\txt_datasets\code\github\menu"

    # 使用一个while循环来迭代每一天
    current_date = start_date
    while current_date < current_date.today():
        urls = [f"https://data.gharchive.org/{current_date.date()}-{i}.json.gz" for i in range(24)]

        for i, url in enumerate(urls):
            name = str(current_date.date()) + "-" + str(i)
            if os.path.exists(dir + f"\\{name}.json"):
                print(dir + f"\\{name}.json", "exists")
                continue
            download_and_extract_json_gz(url, dir + f"\\{name}.json")

        current_date += timedelta(days=1)


def get_github_urls():
    connect = pymysql.connect(user='root', password='root', db='github_urls')
    cur = connect.cursor()
    cur.execute("use github_urls;")

    pths = list(glob(r"G:\txt_datasets\code\github\menu\*.json"))
    # urls = set()

    for pth in pths:
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                urls = set([l for l in re.findall(r"https://github\.com[a-zA-Z0-9/]*?\.git", line) if "commit" not in l and "compare" not in l])
                for url in urls:
                    cur.execute(f"insert IGNORE into urls values('{url}');")
                    connect.commit()



connect = pymysql.connect(user='root', password='root', db='github_urls')
cur = connect.cursor()
cur.execute("use github_urls;")
cur.execute("select * from urls;")


while True:
    result = cur.fetchone()
    if result is None:
        break
    else:
        url = result[0]
        print(url)
        name = url.split('/')[-1][:-4]
        repo_dir = r'G:\txt_datasets\code\github\codes\\' + name
        if os.path.exists(repo_dir):
            print(name, "exists")
            continue
        try:
            os.system(f'"C:\\Program Files\\Git\\bin\\git" clone {url} {repo_dir} 2>./null')
        except:
            print("failed " + url)
        time.sleep(3)



