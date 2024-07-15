import pyarrow.parquet as pq
from glob import glob
import json


pth = r"G:\txt_datasets\wiki"
pths = glob(pth + '\\*.parquet')

menu = []

for pth in pths:
    with pq.ParquetFile(pth) as pf:
        name = pth.split("\\")[-1][:-8]
        txt = pf.read(['text'])[0]
        with open(r"G:\txt_datasets\wiki\txt\{}.txt".format(name), "a", encoding='utf-8') as f:
            for line in txt:
                f.write(str(line))
        print(name)


