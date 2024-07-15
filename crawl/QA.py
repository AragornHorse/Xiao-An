import os
from tqdm import tqdm
import pyarrow.parquet as pq
import json
from glob import glob


def parquet_to_txt():
    pth = r"G:\txt_datasets\QA\validation-00000-of-00001.parquet"
    tar_pth = r"G:\txt_datasets\QA\validation-00000-of-00001.txt"

    c2i = {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'a': 0, 'b': 1, 'c': 2, 'd': 3, 'E': 4, 'e': 4, '1': 0, '2': 1, '3': 2, '4': 3
    }

    txt = """"""

    with pq.ParquetFile(pth) as pf:
        data = pf.read()

        qs = data['question']
        ans = data['answerKey']
        cs = data['choices']

        for i in range(len(data)):
            q = qs[i]
            an = str(ans[i])
            c = cs[i]

            line = f"{q} "
            for i in range(len(c[0])):
                line += f"{c[1][i]}: {c[0][i]} "
            line = line[:-1]
            if line[-1] != '.':
                line += '.'
            line += f" Answer: {an}.\n\n"
            txt += line

    with open(tar_pth, 'w', encoding='utf-8') as f:
        f.write(txt)


pths = glob(r"G:\txt_datasets\QA\dat2a\data\*.jsonl")


for pth in tqdm(pths):

    name = pth.split("\\")[-1][:-6]

    with open(r"G:\txt_datasets\QA\dat2a\data\{}.txt".format(name), 'a', encoding='utf-8') as tf:
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['language'] == 'en':
                    line = f"{data['context']}\n{data['input']}\n"
                    for an in data['answers']:
                        line += an + ', '
                    line = line[:-2]
                    tf.write(line)
                    tf.write('\n\n')

























