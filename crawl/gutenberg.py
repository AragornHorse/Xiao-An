import requests
import time
import random
from threading import Thread
from glob import glob


url = r"https://www.gutenberg.org/cache/epub/{}/pg{}.txt"
root = r"D:\txt_datasets\gutenberg\init"

header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'}

lsts = glob(root + "\\*.txt")
lsts = [int(t.split('.')[0].split('\\')[-1]) for t in lsts]


class Get(Thread):
    def __init__(self, idx):
        super(Get, self).__init__()
        self.idx = idx

    def run(self):
        for i in self.idx:
            if i in lsts:
                continue
            try:
                time.sleep(random.random() * 2)
                u = url.format(i, i)
                pth = root + "\\{}.txt".format(i)

                txt = requests.get(u, headers=header)

                with open(pth, "w", encoding='utf-8') as f:
                    f.write(txt.text)
                print(f"{i} ok")
            except:
                print(f"{i} not ok")


for i in range(900):
    Get(range(i * 2 + 72000, i * 2 + 2 + 72000)).start()


