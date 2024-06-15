import time
import os
import requests
import re
from threading import Thread
from glob import glob

lsts = glob(r"D:\txt_datasets\c4\*.gz")
al_names = [n.split("\\")[-1] for n in lsts]

txt = """"""

with open(r"./c4.html", 'r', encoding='utf-8') as f:
    for line in f:
        txt += line

header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}


class Get(Thread):
    def __init__(self, pths):
        super(Get, self).__init__()
        self.pths = pths

    def run(self):
        for _ in range(50):
            for name in self.pths:
                if name in al_names:
                    continue
                try:
                    time.sleep(5)
                    if '../' in name:
                        continue

                    url = r"https://the-eye.eu/public/AI/STAGING/c4/en/" + name
                    print(url)

                    response = requests.get(url, stream=True, headers=header)
                    if response.status_code == 200:
                        with open(r"D:\txt_datasets\c4\{}".format(name), 'wb') as f:
                            for chunk in response.iter_content(1024):
                                if chunk:
                                    f.write(chunk)
                    print(r"D:\txt_datasets\c4\{}".format(name))
                    al_names.append(name)
                except:
                    os.remove(r"D:\txt_datasets\c4\{}".format(name))
                    print(name, "failed")
            time.sleep(30)


names = []

for line in re.finditer(r"<a href=\"(.*?)\">", txt):
    names.append(line.groups()[0])

names = names[2:]

# print(names)
# print(al_names)

num = 200

for i in range(len(names) // num + 1):
    time.sleep(1)
    Get(names[i * num: min([(i + 1) * num, len(names)])]).start()
