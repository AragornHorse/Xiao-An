import torch
import random
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
import json
from glob import glob


health_care_magic_pth = r"G:\txt_datasets\sft_dataset\HealthCareMagic"              # 131 M
databricks_dolly_pth = r"G:\txt_datasets\sft_dataset\databricks_dolly_15k\format"   # 11.7M
instruct_data_pth = r"G:\txt_datasets\sft_dataset\open_instruct\format"             # 88.3M
open_hermes_3 = r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00003"                  # 447M
open_hermes_2 = r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00002"                  # 391M
open_hermes_1 = r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00001"                  # 315M
open_hermes_0 = r"G:\txt_datasets\sft_dataset\OpenHermes2.5\00000"                  # 392M
financial_instruct_pth = r"G:\txt_datasets\sft_dataset\financial_instruction_aq22\format"   # 240M
deita10k_train_0 = r"G:\txt_datasets\sft_dataset\deita10k\train_0"                  # 344M   ch
deita10k_train_1 = r"G:\txt_datasets\sft_dataset\deita10k\train_1"                  # 331M   ch
deita10k_test_0 = r"G:\txt_datasets\sft_dataset\deita10k\test_0"                    # 17.4M   ch
deita10k_test_1 = r"G:\txt_datasets\sft_dataset\deita10k\test_1"                    # 16.7M   ch
auto_cot_train_pth = r"G:\txt_datasets\sft_dataset\auto_CoT\train"                  # 2.78M
auto_cot_val_pth = r"G:\txt_datasets\sft_dataset\auto_CoT\val"                      # 0.339M
chatbot_instruction_test = r"G:\txt_datasets\sft_dataset\chatbot_instruction_prompts\test"     # 24.9M
chatbot_instruction_train = r"G:\txt_datasets\sft_dataset\chatbot_instruction_prompts\train"   # 99.7M
cot_reformatted_0 = r"G:\txt_datasets\sft_dataset\CoT_reformatted\00000"            # 559M
cot_reformatted_1 = r"G:\txt_datasets\sft_dataset\CoT_reformatted\00001"            # 614M
cot_reformatted_2 = r"G:\txt_datasets\sft_dataset\CoT_reformatted\00002"            # 689M
cot_reformatted_3 = r"G:\txt_datasets\sft_dataset\CoT_reformatted\00003"            # 865M
cot_reformatted_4 = r"G:\txt_datasets\sft_dataset\CoT_reformatted\00004"            # 665M
tofu_pth = r"G:\txt_datasets\sft_dataset\TOFU\format"                               # 4.44M
open_instruct_pth = r"G:\txt_datasets\sft_dataset\open_instruct\format"             # 88.3M
chat_pth = r"G:\txt_datasets\sft_dataset\chat"                                      # 99.3M


class SingleJsonlMenuDataset(Dataset):
    def __init__(self,
                 pth, max_len=512, tokenizer=Tokenizer(), input_tag="[user]", output_tag="[agent]",
                 device=torch.device("cuda")
                 ):
        with open(pth + "\\menu.json", 'r', encoding='utf-8') as f:
            self.menu = json.load(f)
        self.jsonl_pth = glob(pth + "\\*.jsonl")[0]
        self.tokenizer = tokenizer
        self.max_len = max_len + 1
        self.input_tag = input_tag
        self.output_tag = output_tag
        self.device = device

    def __len__(self):
        return len(self.menu)

    def __getitem__(self, item):
        # read a line
        begin = 0 if item == 0 else self.menu[item - 1] + 2
        with open(self.jsonl_pth, 'rb') as f:
            f.seek(begin, 0)
            line = f.readline().decode('utf-8')
        line = json.loads(line)

        # input and output sentences
        ipt = f"{self.input_tag} {line['input']} {self.output_tag} "
        opt = f"{line['output']} {self.input_tag}"

        # input is too lone, just keep the tail
        if len(ipt) > 3 * self.max_len:
            ipt = ipt[-3 * self.max_len:]

        # output is too long, randomly choice a subsequence
        if len(opt) > 5 * self.max_len:

            # choice the head
            start = random.randint(max([-len(ipt), -2 * self.max_len]), len(opt) - 5 * self.max_len)

            # no input
            if start >= 0:
                for i in range(20):
                    if opt[start] == ' ':
                        break
                    start += 1
                opt_token = self.tokenizer.encode(opt, token_num=self.max_len)[:self.max_len]
                x = opt_token[:-1]
                y = opt_token[1:]
                idx = list(range(len(opt_token) - 1))
            else:
                # start should be a word
                while start < 0:
                    if ipt[start] == ' ':
                        break
                    start += 1

                # chosen input
                ipt = ipt[start:]
                ipt_token = self.tokenizer.encode(ipt)

                # cat output into input
                opt_token = self.tokenizer.encode(opt, self.max_len - len(ipt_token))
                idx = list(range(len(ipt_token) - 1, self.max_len - 1))
                tokens = (ipt_token + opt_token)[:self.max_len]
                x = tokens[:-1]
                y = tokens[1:]

        # all the input and output, cat them
        else:
            ipt_token = self.tokenizer.encode(ipt)
            opt_token = self.tokenizer.encode(opt, self.max_len - len(ipt_token))
            tokens = (ipt_token + opt_token)[:self.max_len]
            idx = list(range(len(ipt_token) - 1, len(tokens) - 1))
            x = tokens[:-1]
            y = tokens[1:]

        # isn't max_len, maybe need padding for patching batch
        if len(x) < self.max_len - 1:

            # choose the next line to upper IO
            begin = self.menu[item] + 2 if item < len(self.menu) - 1 else random.choice(self.menu[:-1]) + 2
            with open(self.jsonl_pth, 'rb') as f:
                f.seek(begin, 0)
                line = f.readline().decode('utf-8')

            # just using output
            pad = self.tokenizer.encode(json.loads(line)['output'], self.max_len - len(x) + 10)
            if len(pad) < 50:
                pad += self.tokenizer.encode(json.loads(line)['input'], self.max_len - len(x) + 10)
                if len(pad) < 50:
                    pad += x
                    if len(pad) < 10:
                        return self.__getitem__(random.randint(0, self.__len__() - 1))

            # padding is not long enough
            while len(pad) < self.max_len - len(x):
                pad = pad + pad[:(self.max_len - len(x))]
        else:
            pad = [0]

        return (
            torch.tensor(x, dtype=torch.long, device=self.device),
            torch.tensor(y, dtype=torch.long, device=self.device),
            idx,
            torch.tensor(pad, dtype=torch.long, device=self.device),
        )


def collate_fn_with_pad(batch):
    """
        using the longest sequence as the batch length
        just padding x
    """
    x, y, idx, pad = zip(*batch)
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

    del x, y, pad
    return X, Y, list(idx)


def collate_fn_without_pad(batch):
    """
        using the longest sequence as the batch length
        just padding x
    """
    x, y, idx = zip(*batch)
    X = torch.stack(x)
    Y = torch.stack(y)
    del x, y
    return X, Y, list(idx)


class Sequences:
    def __init__(self):
        self.seqs = []
        self.is_idx = []
        self.next_idx = -1

    def add_to_tail(self, seq: list, need_idx: bool):
        self.seqs += seq
        self.is_idx += [need_idx] * len(seq)
        self.next_idx = -len(seq)

    def add_to_head(self, seq: list, need_idx: bool):
        if len(self.seqs) == 0:
            self.next_idx = -len(seq)
        self.seqs = seq + self.seqs
        self.is_idx = [need_idx] * len(seq) + self.is_idx

    def get_xy_and_idx(self, max_len=None, random_choice=False, keep_tail=False):
        if max_len is None or max_len >= self.__len__():
            idx = [i for i, v in enumerate(self.is_idx[1:]) if v]
            x = self.seqs[:-1]
            y = self.seqs[1:]
        else:
            if random_choice:
                head = random.randint(max([-len(self.seqs), self.next_idx - max_len + 1]), -max_len - 1)
                idx = [i for i, v in enumerate(self.is_idx[head + 1: head + max_len]) if v]
                x = self.seqs[head: head + max_len - 1]
                y = self.seqs[(head + 1): (head + max_len)]
            else:
                if not keep_tail:
                    idx = [i for i, v in enumerate(self.is_idx[1:max_len]) if v]
                    x = self.seqs[:max_len-1]
                    y = self.seqs[1:max_len]
                else:
                    idx = [i for i, v in enumerate(self.is_idx[1-max_len:]) if v]
                    x = self.seqs[-max_len:-1]
                    y = self.seqs[1-max_len:]

        return x, y, idx

    def __len__(self):
        return len(self.seqs)


class JsonlMenuDataset(Dataset):
    def __init__(self,
                 pth, max_len=512, tokenizer=Tokenizer(), input_tag="[user]", output_tag="[agent]",
                 device=torch.device("cuda")
                 ):
        with open(pth + "\\menu.json", 'r', encoding='utf-8') as f:
            self.menu = json.load(f)
        self.jsonl_pth = glob(pth + "\\*.jsonl")[0]
        self.tokenizer = tokenizer
        self.max_len = max_len + 1
        self.input_tag = input_tag
        self.output_tag = output_tag
        self.device = device

    def __len__(self):
        return len(self.menu)

    def __getitem__(self, item):

        seq = Sequences()

        # read a line
        begin = 0 if item == 0 else self.menu[item - 1] + 2
        with open(self.jsonl_pth, 'rb') as f:
           f.seek(begin, 0)
           line = f.readline().decode('utf-8')
        line = json.loads(line)

        # input and output sentences
        ipt = f"{line['input']} {self.output_tag} "
        opt = f"{line['output']} {self.input_tag}"

        # input is too lone, just keep the tail
        if len(ipt) > 3 * self.max_len:
            ipt = ipt[-3 * self.max_len:]

        seq.add_to_tail(self.tokenizer.encode(opt, self.max_len), True)
        seq.add_to_head(self.tokenizer.encode(ipt), False)

        if len(seq) < self.max_len:
            if (random.random() < 0.5 or item < 4) and item < self.__len__() - 4:  # add to tail
                seq.add_to_head(self.tokenizer.encode(self.input_tag), False)
                i = 0
                while len(seq) < self.max_len:
                    begin = self.menu[item + i] + 2
                    i += 1
                    with open(self.jsonl_pth, 'rb') as f:
                        f.seek(begin, 0)
                        line = f.readline().decode('utf-8')
                    line = json.loads(line)
                    seq.add_to_tail(
                        self.tokenizer.encode(
                            f" {line['input']} {self.output_tag}",
                            self.max_len - len(seq)
                        ),
                        False
                    )
                    if len(seq) >= self.max_len:
                        break
                    seq.add_to_tail(
                        self.tokenizer.encode(
                            f"{line['output']} {self.input_tag}",
                            self.max_len - len(seq)
                        ),
                        True
                    )
                x, y, idx = seq.get_xy_and_idx(self.max_len, keep_tail=False)
            else:    # add to head
                i = -2
                while len(seq) < self.max_len:
                    begin = self.menu[item + i] + 2
                    i -= 1
                    with open(self.jsonl_pth, 'rb') as f:
                        f.seek(begin, 0)
                        line = f.readline().decode('utf-8')
                    line = json.loads(line)
                    seq.add_to_head(
                        self.tokenizer.encode(
                            f" {line['output']} {self.input_tag} "
                        ),
                        True
                    )
                    if len(seq) >= self.max_len:
                        break
                    seq.add_to_head(
                        self.tokenizer.encode(
                            f"{line['input']} {self.output_tag} "
                        ),
                        False
                    )
                x, y, idx = seq.get_xy_and_idx(self.max_len, keep_tail=True)
        else:
            seq.add_to_head(self.tokenizer.encode(self.input_tag), False)
            x, y, idx = seq.get_xy_and_idx(self.max_len, random_choice=True)

        return (
           torch.tensor(x, dtype=torch.long, device=self.device),
           torch.tensor(y, dtype=torch.long, device=self.device),
           idx
        )


if __name__ == '__main__':
    from tqdm import tqdm

    # dataset = SingleJsonlMenuDataset(health_care_magic_pth, max_len=500, device=torch.device("cpu"))
    # dataset = SingleJsonlMenuDataset(instruct_data_pth, max_len=500, device=torch.device("cpu"))
    # tk = dataset.tokenizer

    # loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    #
    # for batch in tqdm(loader):
    #     # x, y, idx = batch
    #     # print(x.shape, y.shape)
    #     # x = x[0]
    #     # y = y[0]
    #     # print('-' * 50)
    #     # print(tk.decode(x))
    #     # print("\n")
    #     # print(tk.decode(y))
    #     # print("\n")
    #     # print(tk.decode(y[idx[0]]))
    #     # input(">>>")
    #     pass

    import dataset as D

    block_size = 400
    device = torch.device("cpu")

    dataset = D.RandomMultiDataset([
        {
            'dataset': D.RandomMultiDataset([
                {
                    'dataset': D.TxtFolderDataset(D.c4_txt, max_len=block_size, device=device),  # 741 G
                    'rate': 1.
                }, {
                    'dataset': D.TxtFolderDataset(D.s2orc_txt, max_len=block_size, device=device),  # 577 G
                    'rate': 0.55
                }, {
                    'dataset': D.TxtFolderDataset(D.python_txt, max_len=block_size, device=device),  # 72 G
                    'rate': 0.08
                }, {
                    'dataset': D.TxtFolderDataset(D.books3_txt, max_len=block_size, device=device),  # 144 G
                    'rate': 0.45
                }, {
                    'dataset': D.TxtFolderDataset(D.qa_txt, max_len=block_size, device=device),  # 0.2 G
                    'rate': 0.05
                }, {
                    'dataset': D.TxtFolderDataset(D.wiki_txt, max_len=block_size, device=device),  # 19 G
                    'rate': 0.2
                }, {
                    'dataset': D.TxtFolderDataset(D.gutenberg_txt, max_len=block_size, device=device),  # ~10 G
                    'rate': 0.07
                }, {
                    'dataset': D.TxtFolderDataset(D.LoC_txt, max_len=block_size, device=device),  # 44 G
                    'rate': 0.15
                }, {
                    'dataset': D.TxtFolderDataset(D.email_txt, max_len=block_size, device=device),  # 0.09 G
                    'rate': 0.07
                }
            ], need_idx=True),
            'rate': 0.
        }, {
            'dataset': JsonlMenuDataset(
                health_care_magic_pth, block_size, device=device
            ),
            'rate': 0.13
        }, {
            'dataset': JsonlMenuDataset(
                databricks_dolly_pth, block_size, device=device
            ),
            'rate': 0.01
        }, {
            'dataset': JsonlMenuDataset(
                instruct_data_pth, block_size, device=device
            ),
            'rate': 0.09
        }, {
            'dataset': JsonlMenuDataset(
                open_hermes_3, block_size, device=device
            ),
            'rate': 0.447
        }, {
            'dataset': JsonlMenuDataset(
                open_hermes_2, block_size, device=device
            ),
            'rate': 0.391
        }, {
            'dataset': JsonlMenuDataset(
                open_hermes_1, block_size, device=device
            ),
            'rate': 0.315
        }, {
            'dataset': JsonlMenuDataset(
                open_hermes_0, block_size, device=device
            ),
            'rate': 0.392
        }, {
            'dataset': JsonlMenuDataset(
                financial_instruct_pth, block_size, device=device
            ),
            'rate': 0.24
        }, {
            'dataset': JsonlMenuDataset(
                auto_cot_train_pth, block_size, device=device
            ),
            'rate': 0.003
        }, {
            'dataset': JsonlMenuDataset(
                auto_cot_val_pth, block_size, device=device
            ),
            'rate': 0.0004
        }, {
            'dataset': JsonlMenuDataset(
                chatbot_instruction_test, block_size, device=device
            ),
            'rate': 0.0249
        }, {
            'dataset': JsonlMenuDataset(
                chatbot_instruction_train, block_size, device=device
            ),
            'rate': 0.0997
        }, {
            'dataset': JsonlMenuDataset(
                cot_reformatted_0, block_size, device=device
            ),
            'rate': 0.559 / 2
        }, {
            'dataset': JsonlMenuDataset(
                cot_reformatted_1, block_size, device=device
            ),
            'rate': 0.614 / 2
        }, {
            'dataset': JsonlMenuDataset(
                tofu_pth, block_size, device=device
            ),
            'rate': 0.004
        }, {
            'dataset': JsonlMenuDataset(
                open_instruct_pth, block_size, device=device
            ),
            'rate': 0.088
        }, {
            'dataset': JsonlMenuDataset(
                chat_pth, block_size, device=device
            ),
            'rate': 0.0993
        }
    ], need_idx=True)
    tk = dataset.tokenizer

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_without_pad)

    for (x, y, idx) in tqdm(loader):
        print(x.shape, y.shape, len(idx))
        x = x[0]
        y = y[0]
        idx = idx[0]
        print("-" * 50)
        print(tk.decode(x))
        print("-" * 10)
        print(tk.decode(y))
        print('-' * 10)
        print(tk.decode(y[idx]))
        input(">>>")
        pass



