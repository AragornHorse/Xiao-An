import torch
from torch.utils.data import Dataset
from glob import glob
import json
import random
import pickle as pkl
import tokenizer
import os
from tqdm import tqdm


random.seed(2)


books3_txt = r"G:\txt_datasets\books\txt"
c4_txt = r"G:\txt_datasets\c4\txts"
python_txt = r"G:\txt_datasets\code\python\txt"
gutenberg_txt = r"G:\txt_datasets\gutenberg\pure"
LoC_txt = r"G:\txt_datasets\LoC_PD_Books\txt"
qa_txt = r"G:\txt_datasets\QA\dat2a\txt"
s2orc_txt = r"G:\txt_datasets\s2orc\txt"
wiki_txt = r"G:\txt_datasets\wiki\txt"
chat_txt = r"G:\txt_datasets\chat"
email_txt = r"G:\txt_datasets\emails"


class TxtDataset(Dataset):
    def __init__(self,
                 pth=r"D:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\dialogues_text.txt",
                 tokenizer=tokenizer.Tokenizer(), max_len=512, device=torch.device("cpu"), name=None
                 ):
        """
            load all .txt into memary
        """
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
    def __init__(self, datasets: list, need_idx=False):   # [{dataset: xxx, rate: xxx}]
        self.datasets = datasets
        norm = 0
        for d in self.datasets:
            norm += d['rate']
        for d in self.datasets:
            d['rate'] /= norm
        self.tokenizer = self.datasets[0]['dataset'].tokenizer
        self.need_idx = need_idx

    def __getitem__(self, item):
        global select_dataset_time
        ds_idx = random.random()
        d = {}
        for d in self.datasets:
            if d['rate'] >= ds_idx:
                break
            else:
                ds_idx -= d['rate']
        d_id = random.randint(0, len(d['dataset']) - 1)
        rst = d['dataset'][d_id]
        if self.need_idx:
            if len(rst) == 2:
                x, y = rst
                idx = list(range(x.shape[0]))
                return x, y, idx
            else:
                return rst
        else:
            if len(rst) == 3:
                x, y, idx = rst
                return x, y
            else:
                return rst

    def __len__(self):
        return 200000000


class ArxivTokenDataset(Dataset):
    def __init__(self, tokens_pth=r"G:\txt_datasets\archive (3)\tokens", tokenizer=tokenizer.Tokenizer(),
                 max_len=512, device=torch.device("cpu")):
        """
            tokens
        """
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


class TxtFolderDataset(Dataset):
    def __init__(self, pth=r"G:\txt_datasets\gutenberg\pure",
                 tokenizer=tokenizer.Tokenizer(), max_len=512, device=torch.device("cpu"), name=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device
        self.name = name
        if os.path.isdir(pth):
            self.pths = glob(pth + r"\\*.txt")
        else:
            with open(pth, "r", encoding="utf-8") as f:
                self.menu = json.load(f)
            self.pths = list(self.menu.keys())

    def __len__(self):
        return len(self.pths)

    def __getitem__(self, item):

        # chosen file
        pth = self.pths[item]

        # byte length
        length = os.path.getsize(pth)

        # randomly read by byte
        start = random.randint(0, max(1, length - self.max_len * 4))
        with open(pth, 'rb') as f:
            try:

                f.seek(start, 0)

                # read long bytes, may cause error
                line = f.read(min(self.max_len * 5, length - start)).decode('utf-8')

                # the first token should be a word
                i = line[:20].find(' ')
                i = i if i > 0 else 0

                tokens = self.tokenizer.encode(line[i+1:], self.max_len + 1)

            # start isn't a beginning of a charactor, causing failed to decode in utf-8
            except UnicodeDecodeError:
                # print(1)
                return self.__getitem__(random.randint(0, len(self.pths) - 1))

        # token number is not enough
        if len(tokens) < self.max_len + 1:
            # print(2)
            return self.__getitem__(random.randint(0, len(self.pths) - 1))

        return [
            torch.tensor(tokens[:-1], dtype=torch.long, device=self.device),
            torch.tensor(tokens[1:], dtype=torch.long, device=self.device)
        ]



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

    # da = ArxivTokenDataset()
    # # print(da[len(da) - 2])
    # for i, d in enumerate(da):
    #     print(f"{i} / {len(da)}")
    #     x, y = d
    #     print(da.tokenizer.decode(x))
    #     print(da.tokenizer.decode(y))
    #     input()

    data = RandomMultiDataset([
        {
            'dataset': TxtFolderDataset(c4_txt),     # 741 G
            'rate': 1.
        }, {
            'dataset': TxtFolderDataset(s2orc_txt),    # 577 G
            'rate': 0.5
        }, {
            'dataset': TxtFolderDataset(python_txt),   # 72 G
            'rate': 0.05
        }, {
            'dataset': TxtFolderDataset(books3_txt),    # 144 G
            'rate': 0.3
        }, {
            'dataset': TxtFolderDataset(qa_txt),    # 0.2 G
            'rate': 0.05
        }, {
            'dataset': TxtFolderDataset(wiki_txt),   # 19 G
            'rate': 0.1
        }, {
            'dataset': TxtFolderDataset(gutenberg_txt),     # ~10 G
            'rate': 0.08
        }, {
            'dataset': TxtFolderDataset(LoC_txt),       # 44 G
            'rate': 0.15
        }, {
            'dataset': TxtFolderDataset(chat_txt),   # 0.05 G
            'rate': 0.05
        }
    ])

    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    for d in tqdm(loader):
        x, y = d
        # print(x.shape, y.shape)
        # print(data.tokenizer.decode(x[0]))
        # print('-' * 20)
        # print(data.tokenizer.decode(y[0]))
        # print("\n" * 4)

