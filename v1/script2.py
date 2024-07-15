import json
import re
from glob import glob
import pymysql
from langdetect import detect
import gzip
from tqdm import tqdm


def D2G():
    pth = r"G:\txt_datasets\gutenberg\pure\menu.json"

    with open(pth, 'r', encoding='utf-8') as f:
        menu = json.load(f)

    menu2 = {}

    for k in menu:
        menu2['G' + k[1:]] = menu[k]

    with open(pth, 'w', encoding='utf-8') as f:
        json.dump(menu2, f)


def check_s2orc():
    with open(r"G:\txt_datasets\s2orc\20240621_112401_00086_9cfub_0003fb43-f611-4092-8f5a-fd378e816c27\20240621_112401_00086_9cfub_0003fb43-f611-4092-8f5a-fd378e816c27", 'r', encoding='utf-8') as f:
        for line in f:
            print(line)

    pth = r"G:\txt_datasets\s2orc\20240621_112401_00086_9cfub_0ee2d605-207e-4308-8f6b-c5b299a6754f\20240621_112401_00086_9cfub_0ee2d605-207e-4308-8f6b-c5b299a6754f"
    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # dict_keys(['corpusid', 'externalids', 'content'])
            # dict_keys(['source', 'text', 'annotations'])
            print(data['content']['text'])
            break


def s2orc_to_txt():
    pths = glob(r"G:\txt_datasets\s2orc\*.gz")

    for i, pth in enumerate(pths):
        with gzip.open(pth, 'rt', encoding='utf-8') as f:
            name = pth.split('\\')[-1][:-3]
            with open(r"G:\txt_datasets\s2orc\txt\\" + name + ".txt", 'a', encoding='utf-8') as tf:
                for line in tqdm(f):
                    line = json.loads(str(line))
                    ctn = line['content']['text']
                    if ctn is None:
                        continue
                    tf.write(ctn)
                    tf.write('\n\n')
            print(i)


def chat_to_jsonl():
    pths = [
        r"D:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\format.txt",
        r"D:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\format.txt",
        r"D:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\format.txt",
        r"D:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\format.txt",
        r"D:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\format.txt",
        r"D:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\format.txt"
    ]


    with open(r"G:\txt_datasets\sft_dataset\chat\train.jsonl", 'a', encoding='utf-8') as tf:
        for pth in pths:
            with open(pth, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line[6:-1]
                    line = re.split(r"\[user\]|\[agent\]", line)
                    line.remove('')
                    for i in range(len(line) // 2):
                        ipt = line[2 * i]
                        opt = line[2 * i + 1]

                        rst = {
                            'input': ipt,
                            'output': opt
                        }

                        json.dump(rst, tf)
                        tf.write('\n')

        for pth in pths:
            with open(pth, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line[6:-1]
                    line = re.split(r"\[user\]|\[agent\]", line)
                    line.remove('')
                    for i in range((len(line) - 1) // 2):
                        ipt = line[2 * i + 1]
                        opt = line[2 * i + 2]

                        rst = {
                            'input': ipt,
                            'output': opt
                        }

                        json.dump(rst, tf)
                        tf.write('\n')

