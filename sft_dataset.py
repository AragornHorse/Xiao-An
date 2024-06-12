from torch.utils.data import Dataset, DataLoader
import tokenizer as Tokenizer
import torch
import random


class ChatDataset(Dataset):
    def __init__(self, pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\format.txt",
                 tokenizer=Tokenizer.Tokenizer(), max_len=512, device=torch.device("cpu"), pretrain_rate=0.05,
                 name=None):
        self.pth = pth
        self.name = name
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lst = []
        self.chat = tokenizer.encode('[chat]')
        self.agent = tokenizer.encode('[agent]')
        self.usr = tokenizer.encode('[user]')
        self.pretrain_rate = pretrain_rate

        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                line = self.tokenizer.encode(line)
                if 15 < len(line) < 2500:
                    if len(line) <= self.max_len + 1:
                        self.lst.append(line)
                    else:
                        lst = []
                        end = len(self.chat) + len(self.usr) + 1
                        while end < len(line):
                            while line[end - len(self.usr): end] != self.usr:
                                end += 1
                                if end >= len(line):
                                    break
                            start = max([0, end - self.max_len - 1])
                            lst.append(line[start: min([end, len(line)])])
                            end += 1
                        for i in range(len(lst) - 1):
                            if lst[i][:len(self.chat)] != self.chat:
                                break
                        self.lst += lst[i-1:]

        print(f"{self.name} ready")

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, item):
        lst = self.lst[item]
        lst_ = [lst[-1]]
        i = 0
        while len(lst_) + len(lst) < self.max_len + 5:
            lst_ += self.lst[(item + i) % self.__len__()]
            i += 1
        x = torch.tensor(lst[:-1], dtype=torch.long, device=self.device)
        y = torch.tensor(lst[1:], dtype=torch.long, device=self.device)
        pad = torch.tensor(lst_, dtype=torch.long, device=self.device)
        if random.random() > self.pretrain_rate:
            idx = self.get_idx(lst)
        else:
            idx = list(range(y.shape[0]))
        return x, y, pad, idx

    def get_idx(self, lst):
        idx = []
        is_agent = False
        if lst[:len(self.chat)] == self.chat:    # all agent word
            for i in range(len(self.chat), len(lst) - 1):
                now_id = i + 1
                if lst[max([0, now_id - len(self.agent)]): now_id] == self.agent:
                    is_agent = True
                if lst[max([0, now_id - len(self.usr)]): now_id] == self.usr:
                    is_agent = False
                if is_agent:
                    idx.append(i)
        else:    # just the last sen
            for i in reversed(range(len(lst))):
                idx.append(i - 1)
                if lst[max(0, i - len(self.agent)): i] == self.agent:
                    break
            idx = list(reversed(idx))

        return idx


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


def collate_fn(batch):
    x, y, pad, idx = zip(*batch)
    device = x[0].device
    max_len = max([_.shape[0] for _ in y])
    batch_size = len(x)
    X = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    Y = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i in range(batch_size):
        X[i, :len(x[i])] = x[i]
        Y[i, :len(y[i])] = y[i]
        if len(x[i]) < max_len:
            X[i, len(x[i]):] = pad[i][:max_len - len(x[i])]

    del x, y
    return X, Y, list(idx)


if __name__ == '__main__':

    d = MultiDataset([
        ChatDataset(
            pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\format.txt",
            max_len=384,
            pretrain_rate=0.05,
            name="IJCNLPDailyDialog"
        ),
        ChatDataset(
            pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\format.txt",
            max_len=384,
            pretrain_rate=0.05,
            name="CMUDoG"
        ),
        ChatDataset(
            pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\format.txt",
            max_len=384,
            pretrain_rate=0.05,
            name="blended_skill_talk"
        ),
        ChatDataset(
            pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\format.txt",
            max_len=384,
            pretrain_rate=0.05,
            name="woz_dialogs"
        ),
        ChatDataset(
            pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\format.txt",
            max_len=384,
            pretrain_rate=0.05,
            name="daily dialog"
        ),
        ChatDataset(
            pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\format.txt",
            max_len=384,
            pretrain_rate=0.05,
            name='empathetic dialogues'
        )
    ])

    loader = DataLoader(d, batch_size=48, shuffle=True, collate_fn=collate_fn)

    for data in loader:
        x, y, idx = data

        for i in range(32):
            print('=' * 20)
            print(d.datasets[0].tokenizer.decode(x[i]))
            print('-' * 20)
            # print(d.tokenizer.decode(y[i]))
            print(d.datasets[0].tokenizer.decode(y[i, idx[i]]))
            print('=' * 20)




