import tokenizer as Tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
import random
import pickle as pkl


class IJCNLPDailyDialog(Dataset):
    def __init__(self,
                 pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\dialogues_text.txt",
                 tokenizer=Tokenizer.Tokenizer(), max_len=512, device=torch.device("cpu"), name=None
                 ):
        print(f"reading {name}")
        self.txt = """"""
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                self.txt += line
        self.txt = tokenizer.encode(self.txt)
        self.device = device
        self.max_len = max_len

    def __getitem__(self, item):
        return [
            torch.tensor(self.txt[item: item + self.max_len], dtype=torch.long, device=self.device),
            torch.tensor(self.txt[item + 1: item + self.max_len + 1], dtype=torch.long, device=self.device)
        ]

    def __len__(self):
        return len(self.txt) - self.max_len - 1


class CMUDoG(Dataset):
    def __init__(self,
                 pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\all.txt",
                 tokenizer=Tokenizer.Tokenizer(), max_len=512, device=torch.device("cpu")):
        print("reading CMUDoG")
        self.txt = """"""
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                self.txt += line.replace("<", "").replace(">", "")
        self.txt = tokenizer.encode(self.txt)
        self.device = device
        self.max_len = max_len

    def get_txt(self, pth):
        print("reading CMUDoG")
        for tp in ['train', 'test', 'valid']:
            dir_pth = pth + "\\" + tp + "\\"
            lst = glob(dir_pth + "*.json")
            for i, data_pth in enumerate(lst):
                with open(data_pth, "r", encoding='utf-8') as f:
                    data = json.load(f)['history']
                now_usr = data[0]['uid']
                self.txt += "\n" + data[0]['text']
                if self.txt[-1] not in ['.', '?', '!']:
                    self.txt += "."
                for line in data:
                    if self.txt[-1] not in ['.', '?', '!']:
                        self.txt += '.'
                    usr = line['uid']
                    if usr == now_usr:
                        self.txt += line['text']
                    else:
                        now_usr = usr
                        self.txt += "__eou__" + line['text']
                print(f"{i} / {len(lst)}")

                if self.txt[-1] not in ['.', '?', '!']:
                    self.txt += '.'
                self.txt += "__eou__"

    def __getitem__(self, item):
        return [
            torch.tensor(self.txt[item: item + self.max_len], dtype=torch.long, device=self.device),
            torch.tensor(self.txt[item + 1: item + self.max_len + 1], dtype=torch.long, device=self.device)
        ]

    def __len__(self):
        return len(self.txt) - self.max_len - 1


class MultiDataset(Dataset):
    def __init__(self, datasets):
        self.datasets: list = datasets

    def __getitem__(self, item):
        for d in self.datasets:
            if item < len(d):
                return d[item]
            else:
                item = item - len(d)

    def __len__(self):
        l = 0
        for d in self.datasets:
            l += len(d)
        return l


class RandomMultiDataset(Dataset):
    def __init__(self, datasets: list):   # [{dataset: xxx, rate: xxx}]
        self.datasets = datasets
        norm = 0
        for d in self.datasets:
            norm += d['rate']
        for d in self.datasets:
            d['rate'] /= norm

    def __getitem__(self, item):
        ds_idx = random.random()
        d = {}
        for d in self.datasets:
            if d['rate'] >= ds_idx:
                break
            else:
                ds_idx -= d['rate']
        d_id = random.randint(0, len(d['dataset']) - 1)
        return d['dataset'][d_id]

    def __len__(self):
        return 1000000


class ArxivAbstract(Dataset):
    def __init__(self, pth=r"D:\txt_datasets\archive (3)\arxiv-metadata-oai-snapshot.json",
                 tokenizer=Tokenizer.Tokenizer(), max_len=512, device=torch.device("cpu")):
        self.txt = []
        now_ab = []
        print("loading arxiv")
        with open(pth, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                abstract = d['abstract'].replace("\n", '')
                title = d['title'].replace("\n", "").replace("  ", " ")
                now_ab += tokenizer.encode(
                    f'[chat] [user] Please write an abstract with title "{title}". [agent] {abstract} \n'
                )
                if len(now_ab) >= max_len:
                    self.txt.append(now_ab)
                    now_ab = []
                if i % 100000 == 0:
                    print(f"{i} / {2400000}")
                # if i > 10:
                #     break
        print("loaded")

        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        item = item % self.__len__()
        for ab in self.txt:
            if len(ab) >= item + self.max_len + 1:
                return torch.tensor(ab[item: item + self.max_len], dtype=torch.long, device=self.device),\
                       torch.tensor(ab[item + 1: item + self.max_len + 1], dtype=torch.long, device=self.device)
            else:
                item = item - len(ab) + self.max_len

    def __len__(self):
        all = 0
        for ab in self.txt:
            all += len(ab) - self.max_len
        return all


class ArxivTokenDataset(Dataset):
    def __init__(self, tokens_pth=r"D:\txt_datasets\archive (3)\tokens", tokenizer=Tokenizer.Tokenizer(),
                 max_len=512, device=torch.device("cpu")):
        self.lst = glob(tokens_pth + r"\*.pkl")
        self.max_len = max_len
        self.device = device
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        pth = self.lst[item]
        with open(pth, 'rb') as f:
            tokens = pkl.load(f)
        frm = random.randint(0, len(tokens) - self.max_len - 1)
        x = tokens[frm: frm + self.max_len]
        y = tokens[frm + 1: frm + 1 + self.max_len]
        x = torch.tensor(x, dtype=torch.long, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        return [x, y]


if __name__ == '__main__':

    # for d in RandomMultiDataset([{
    #         'dataset': IJCNLPDailyDialog(),
    #         'rate': 1
    #     }, {
    #         'dataset': CMUDoG(),
    #         'rate': 0.1
    #     }
    # ]):
    #     print(d)

    da = ArxivTokenDataset()
    # print(da[len(da) - 2])
    for i, d in enumerate(da):
        print(f"{i} / {len(da)}")
        x, y = d
        print(da.tokenizer.decode(x))
        print(da.tokenizer.decode(y))
        input()
