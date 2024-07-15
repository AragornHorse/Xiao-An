import gzip
import os
import shutil
from glob import glob
import json
import re
from tqdm import tqdm


def gutenberg_pure():
    from_dir = r"G:\txt_datasets\gutenberg\init"
    to_dir = r"G:\txt_datasets\gutenberg\pure"

    lst = glob(from_dir + "\\*.txt")

    for pth in lst:

        txt = """"""
        with open(pth, "r", encoding="utf-8") as f:
            flag = False
            for line in f:
                if flag and "*** END OF THE PROJECT GUTENBERG" not in line:
                    txt += line
                if "*** START OF THE PROJECT GUTENBERG EBOOK " in line:
                    flag = True

                if "*** END OF THE PROJECT GUTENBERG" in line:
                    flag = False

        txt = txt.replace("\n\n", " ")
        txt = txt.replace("  ", "\n")

        name = pth.split("\\")[-1]
        with open(to_dir + f"\\{name}", "w", encoding='utf-8') as f:
            f.write(txt)

        print(name)


def gutenberg_generate_menu():
    from langdetect import detect

    dir = r"G:\txt_datasets\gutenberg\pure"

    pths = glob(dir + "\\*.txt")

    menu = {}

    for i, pth in enumerate(pths):

        txt = """"""
        with open(pth, "r", encoding="utf-8") as f:
            for line in f:
                if line[:len("Produced by")] != "Produced by":
                    txt += line

        txt = re.sub("\n+", "\n", txt)
        with open(pth, 'w', encoding="utf-8") as f:
            f.write(txt)

        if (len(txt) > 1000) and detect(txt) == 'en':
            menu[pth] = len(txt)
        print(i, len(menu))

    with open(dir + "\\menu.json", 'w', encoding="utf-8") as f:
        json.dump(menu, f)


def unzip_c4():
    lsts = glob(r"G:\txt_datasets\c4\*.gz")
    print(len(lsts))


    def uncompress_file(gz_filepath, dest_path):
        with gzip.open(gz_filepath, 'rb') as f_in:
            with open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    for pth in lsts:
        name = pth.split("\\")[-1][:-3]
        dest = r"G:\txt_datasets\c4\\" + name
        print(dest)
        if not os.path.exists(dest):
            uncompress_file(pth, dest)


def c4_to_txt():
    pth = r"G:\txt_datasets\c4"

    pths = glob(pth + "\\*.json")


    def find_newline_offsets(filename):
        with open(filename, 'rb') as file:
            offsets = []
            current_offset = 0
            while True:
                byte = file.read(1)

                if not byte:
                    break

                if byte == b'\n' or (byte == b'\r' and file.peek(1) == b'\n'):
                    offsets.append(current_offset)

                    if byte == b'\r' and file.peek(1) == b'\n':
                        file.read(1)
                current_offset += 1

            return offsets


    for i, pth in enumerate(pths):
        if os.path.exists(r"G:\txt_datasets\c4\txts\{}.txt".format(i)):
            print(i, pth)
            continue
        with open(r"G:\txt_datasets\c4\txts\{}.txt".format(i), "a", encoding="utf-8") as tf:
            print(i, pth)
            with open(pth, "r", encoding="utf-8") as f:
                for line in f:
                    tf.write(json.loads(line)['text'] + "\n\n")


def check_Loc():
    import pyarrow.parquet as pq

    pths = glob(r"G:\txt_datasets\LoC_PD_Books\*.parquet")

    for pth in pths:
        try:
            with pq.ParquetFile(pth) as pf:
                txt = pf.read(['text'])[0]
        except:
            print(pth)


def books_par_to_txt():
    pths = glob(r"G:\txt_datasets\LoC_PD_Books\*.parquet")

    import pyarrow.parquet as pq


    for pth in tqdm(pths):
        name = pth.split('\\')[-1][:-8]
        print(name)
        if os.path.exists(r"G:\txt_datasets\LoC_PD_Books\txt\{}.txt".format(name)):
            continue
        with pq.ParquetFile(pth) as pf:
            line = pf.read(['text'])[0]
            with open(r"G:\txt_datasets\LoC_PD_Books\txt\{}.txt".format(name), 'a', encoding='utf-8') as f:
                for l in line:
                    f.write(str(l))


def python_github_to_txt():
    pths = glob(r"G:\txt_datasets\code\python\*.parquet")

    import pyarrow.parquet as pq

    for pth in tqdm(pths):
        name = pth.split('\\')[-1][:-8]
        print(name)
        if os.path.exists(r"G:\txt_datasets\code\python\txt\{}.txt".format(name)):
            continue
        with pq.ParquetFile(pth) as pf:
            line = pf.read(['code'])[0]
            with open(r"G:\txt_datasets\code\python\txt\{}.txt".format(name), 'a', encoding='utf-8') as f:
                for l in line:
                    f.write("```")
                    f.write(str(l))
                    f.write("```")
# python_github_to_txt()


def python_25k():
    pth = r"G:\txt_datasets\code\python-codes-25k.json"

    with open(pth, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(r"G:\txt_datasets\code\python-codes-25k.txt", 'a', encoding='utf-8') as f:
        for line in data:
            f.write(line['text'])
            f.write('\n\n')


def python_train_to_txt():
    pth = r"G:\txt_datasets\code\train.jsonl"

    with open(r"G:\txt_datasets\code\train.txt", 'a', encoding='utf-8') as tf:
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                for sol in eval(data['solutions']):
                    tf.write(f"{data['question']} ```{sol}```. Result is {data['input_output']} \n\n")


def book_to_txt():
    pths = glob(r"G:\txt_datasets\books\*.jsonl")


    for pth in tqdm(pths):

        name = pth.split('\\')[-1][:-6]

        if os.path.exists(r"G:\txt_datasets\books\txt\{}.txt".format(name)):
            continue

        with open(r"G:\txt_datasets\books\txt\{}.txt".format(name), 'a', encoding='utf-8') as tf:
            with open(pth, 'r', encoding='utf-8') as f:
                for line in f:
                    line = json.loads(line)['text']
                    tf.write(line)
                    tf.write('\n\n')


def check_stack_overflow():
    pth = r"G:\txt_datasets\stackoverflow\stackoverflow_0000.jsonl"

    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            print(line['text'])
            input()


def email_to_txt():
    pths = glob(r"G:\txt_datasets\emails\*.parquet")

    import pyarrow.parquet as pq

    for pth in tqdm(pths):
        name = pth.split('\\')[-1][:-8]
        print(name)
        if os.path.exists(r"G:\txt_datasets\emails\{}.txt".format(name)):
            continue
        with pq.ParquetFile(pth) as pf:
            line = pf.read(['text'])[0]
            with open(r"G:\txt_datasets\emails\{}.txt".format(name), 'a', encoding='utf-8') as f:
                for l in line:
                    f.write(str(l))
                    f.write("\n\n")
