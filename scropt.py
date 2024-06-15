from glob import glob
import json
import pandas as pd
import re


def manage_blended_skill_talk():
    pth = r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk"

    data = []

    for p in glob(pth + "\\*.json"):
        with open(p, 'r', encoding='utf-8') as f:
            data += json.load(f)

    txt = """"""

    for d in data:
        d = d['dialog']
        now_usr = d[0][0]
        ctt = d[0][1]
        txt += "\n" + ctt

        for line in d[1:]:
            usr, content = line
            if txt[-1] not in ['.', ',', '?', '!']:
                txt += '.'
            if usr == now_usr:
                txt += content
            else:
                txt += "__eou__" + content
                now_usr = usr
        txt += "__eou__"

    with open(r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\all.txt", 'w', encoding='utf-8') as f:
        f.write(txt)


def view_convai_chitchat():
    pth = r"C:\Users\DELL\Desktop\datasets\supper_replier\Convai_Chitchat"
    pths = glob(pth + "\\*.json")

    data = []

    for pth in pths:
        with open(pth, 'r', encoding='utf-8') as f:
            data += json.load(f)

    for d in data:
        d = d['thread']
        if len(d) < 2:
            continue
        for _ in d:
            print(_['userId'], _['text'])


def woz_dialogs():
    pth = r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs"
    pths = glob(pth + "\\*.json")

    data = []

    for pth in pths:
        with open(pth, 'r', encoding='utf-8') as f:
            data += json.load(f)

    txt = """"""

    for d in data:
        d = d['utterances']

        if len(d) < 2:
            continue

        now_usr = d[0]['speaker']
        ctn = d[0]['text']
        txt += f"\n{ctn}"

        for line in d[1:]:
            if txt[-1] not in ['.', '!', '?', '!']:
                txt += '.'
            usr = line['speaker']
            ctn = line['text']
            if usr == now_usr:
                txt += ctn
            else:
                txt += "__eou__" + ctn
                now_usr = usr

        if txt[-1] not in ['.', '?', '!']:
            txt += '.'
        txt += '__eou__'

    with open(r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\all.txt", 'w', encoding='utf-8') as f:
        f.write(txt)


def daily_dialogs():
    pth = r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog"
    pths = glob(pth + "\\*.json")

    data = []

    txt = """"""

    for pth in pths:
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                data = eval(line)
                data = data['dialogue']
                for d in data:
                    txt += d['text'] + "__eou__"
                txt += "\n"

    with open(r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\all.txt", 'w', encoding='utf-8') as f:
        f.write(txt)


def empathetic_dialogues():
    pth = r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues"
    pths = glob(pth + "\\*.csv")

    txt = """"""

    for pth in pths:

        last_did = 5
        now_uid = '0'

        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = line.split(',')
                    did = d[1]
                    uid = d[4]
                    ctn = d[5]
                    ctn = ctn.replace("_comma_", ",")
                    ctn = ctn.replace("_", " ")

                    if int(did) < last_did:
                        txt += "__eou__\n"
                        txt += ctn
                    else:
                        if uid == now_uid:
                            txt += ctn
                        else:
                            txt += "__eou__" + ctn

                    if txt[-1] not in ['?', '.', '!']:
                        txt += '.'

                    last_did = int(did)
                    now_uid = uid
                except:
                    pass

    with open(r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\all.txt", 'w', encoding='utf-8') as f:
        f.write(txt)


def arxiv_abstract():
    import tokenizer
    tokenizer = tokenizer.Tokenizer()
    pth = r"D:\txt_datasets\archive (3)\arxiv-metadata-oai-snapshot.json"
    with open(pth, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            ab = data['abstract'].replace("\n", '')
            print(data['title'].replace("\n", "").replace("  ", " "))
            print(len(tokenizer.encode(ab)))
            input(">")

# arxiv_abstract()


def eou_to_usr(pth1, pth2):
    txt = """"""
    with open(pth1, "r", encoding='utf-8') as f:
        for line in f:
            l = line.split("__eou__")
            if len(l) < 2:
                continue
            txt += "[chat]"
            for i, s in enumerate(l):
                if i % 2 == 0:
                    usr = "[user]"
                else:
                    usr = "[agent]"
                if s != "\n":
                    txt += f"{usr} {s}"
            txt += "\n"

    with open(pth2, "w", encoding="utf-8") as f:
        f.write(txt)


# eou_to_usr(
#     pth1=r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\all.txt",
#     pth2=r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\format.txt"
# )

# eou_to_usr(
#     pth1=r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\dialogues_text.txt",
#     pth2=r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\format.txt"
# )

# eou_to_usr(
#     pth1=r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\all.txt",
#     pth2=r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\format.txt"
# )

# eou_to_usr(
#     pth1=r"C:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\all.txt",
#     pth2=r"C:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\format.txt"
# )

# eou_to_usr(
#     pth1=r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\all.txt",
#     pth2=r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\format.txt"
# )

# eou_to_usr(
#     pth1=r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\all.txt",
#     pth2=r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\format.txt"
# )

def arxiv_to_txt():
    txt = """"""

    with open(r"D:\txt_datasets\archive (3)\arxiv-metadata-oai-snapshot.json", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            abstract = d['abstract'].replace("\n", ' ')
            title = d['title'].replace("\n", " ").replace("  ", " ")
            txt += f'[chat] [user] "{title}". [agent] {abstract} \n'
            # print('|', d['categories'])
            if (i + 1) % 200 == 0:
                with open(r"D:\txt_datasets\archive (3)\txts\format{}.txt".format(i // 200), 'w', encoding='utf-8') as f:
                    f.write(txt)
                txt = """"""
            if i % 50000 == 0:
                print(f"{i} / {2400000}")

    with open(r"D:\txt_datasets\archive (3)\txts\format.txt", 'w', encoding='utf-8') as f:
        f.write(txt)


def arxiv_txt_to_tokens():
    from glob import glob
    import pickle as pkl
    import tokenizer

    lst = glob(r"D:\txt_datasets\archive (3)\txts\*.txt")
    tokenizer = tokenizer.Tokenizer()

    for pth in lst:
        name = pth.split('\\')[-1].split('.')[0]
        txt = """"""
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                txt += line
        txt = tokenizer.encode(txt)
        with open(r"D:\txt_datasets\archive (3)\tokens\{}.pkl".format(name), 'wb') as f:
            pkl.dump(txt, f)
        print(name)


# arxiv_to_txt()
# arxiv_txt_to_tokens()