import re
import heapq
import json
import pickle as pkl


class Tokenizer:
    def __init__(self, param_pth=r"./tokenizer_param"):
        self.token_to_idx = {}
        self.tokens = []
        self.word_freq = {}
        self.syms = [
            ',', '，', '\.', '\?', '？', '。', '_', '-', '\+', '=', '、', "'",
            '"', '\(', '\)', '\[', '\]', '\*', '%', '$', '\n', '\t'
        ]
        self.word_end_sym = '</w>'
        self.word_to_bpe_cache = {}
        self.load(param_pth)
        self.voc_num = len(self.token_to_idx) + 1

    def bpe(self, txt, add_num_per_iter=100, voc_size=10000):

        # "i am" -> ["i", "am"]
        words = [w for w in re.split(r"|".join(self.syms) + r"| +", txt) if len(w) > 0]

        # {word: freq}
        for word in words:
            if word in self.word_freq.keys():
                self.word_freq[word] += 1
            else:
                self.word_freq[word] = 1
            for c in word:
                if c not in self.tokens:
                    self.tokens.append(c)

        # [a, b, c...]
        self.tokens = self.tokens + [s.replace("\\", "") for s in self.syms] + [self.word_end_sym]

        # [['a', 'm'], freq]
        word_freq = [[[c for c in word] + [self.word_end_sym], self.word_freq[word]] for word in self.word_freq.keys()]

        while len(self.tokens) < voc_size:

            # {"a": {"am": freq}}
            token_pair_freq = {}
            for word, freq in word_freq:
                for i in range(len(word) - 1):
                    c1 = word[i]
                    c2 = word[i+1]
                    if c1 not in token_pair_freq.keys():
                        token_pair_freq[c1] = {}
                    if c2 not in token_pair_freq[c1].keys():
                        token_pair_freq[c1][c2] = freq
                    else:
                        token_pair_freq[c1][c2] += freq

            # [[a, m], [y, o]]
            pairs = []
            for c1 in token_pair_freq.keys():
                pairs += [[c1, c2] for c2 in token_pair_freq[c1].keys()]

            # [[a, m]] to combine
            top = heapq.nlargest(add_num_per_iter, pairs, key=lambda lst: token_pair_freq[lst[0]][lst[1]])
            self.tokens += [pair[0] + pair[1] for pair in top]

            # ['a', 'm'] -> ['am']
            for idx, _ in enumerate(word_freq):
                word, freq = _

                if len(word) <= 1:
                    continue

                for i in range(len(word) - 1):

                    if i >= len(word) - 1:
                        break

                    c1 = word[i]
                    c2 = word[i+1]

                    for pair in top:
                        if c1 == pair[0] and c2 == pair[1]:
                            word[i] = c1 + c2
                            word.pop(i+1)
                            break

                    if c2 == self.word_end_sym:
                        break

                word_freq[idx][0] = word

        # {"am": 1}
        self.token_to_idx = {k: v for v, k in enumerate(self.tokens)}

        # {"work": ["w", "or", "k"]}
        for word, freq in word_freq:
            if freq > 1:
                self.word_to_bpe_cache["".join(word)[:-len(self.word_end_sym)]] = [self.token_to_idx[c] for c in word]

    def encode(self, sen):

        if isinstance(sen, list):
            enc = [self.encode(s) for s in sen]
        else:
            # split words
            words = [w for w in re.split(r"(" + r")|(".join(self.syms) + r")| +", sen) if w is not None and len(w) > 0]
            enc = []

            for i, word in enumerate(words):
                if word in self.word_to_bpe_cache.keys():  # cached word
                    enc += self.word_to_bpe_cache[word]
                else:
                    if word in self.token_to_idx.keys():     # word is token
                        enc.append(self.token_to_idx[word])
                    else:                    # split word
                        word = word + self.word_end_sym      # word</w>
                        start = 0
                        token = []
                        while True:      # select longest token
                            flag = True
                            end = len(word)
                            if end <= start:
                                break
                            while word[start: end] not in self.token_to_idx.keys():
                                end -= 1
                                if end <= start:
                                    flag = False
                                    break
                            if not flag:    # unknown token
                                break
                            token.append(self.token_to_idx[word[start: end]])
                            start = end
                            if start > len(word) - 1:   # finished
                                break
                        enc += token

        return enc

    def decode(self, token):
        if isinstance(token[0], list):
            dec = [self.decode(sen) for sen in token]
        else:
            dec = ""
            for c in token:
                t = self.tokens[c]
                if t[-len(self.word_end_sym):] == self.word_end_sym:
                    dec += t[:-len(self.word_end_sym)] + ' '
                else:
                    dec += t

        return dec

    def save(self, pth):
        with open(pth + "\\" + "tokens.pkl", 'wb') as f:
            pkl.dump(self.tokens, f)
        with open(pth + "\\" + "token2id.json", 'w', encoding='utf-8') as f:
            json.dump(self.token_to_idx, f)
        with open(pth + "\\" + "word_freq.json", 'w', encoding='utf-8') as f:
            json.dump(self.word_freq, f)
        with open(pth + "\\" + "cache.json", 'w', encoding='utf-8') as f:
            json.dump(self.word_to_bpe_cache, f)
        with open(pth + "\\" + "syms.pkl", 'wb') as f:
            pkl.dump(self.syms, f)
        with open(pth + "\\" + "word_end_sym.pkl", 'wb') as f:
            pkl.dump(self.word_end_sym, f)

    def load(self, pth):
        if pth is None:
            return
        with open(pth + "\\" + "tokens.pkl", 'rb') as f:
            self.tokens = pkl.load(f)
        with open(pth + "\\" + "token2id.json", 'r', encoding='utf-8') as f:
            self.token_to_idx = json.load(f)
        with open(pth + "\\" + "word_freq.json", 'r', encoding='utf-8') as f:
            self.word_freq = json.load(f)
        with open(pth + "\\" + "cache.json", 'r', encoding='utf-8') as f:
            self.word_to_bpe_cache = json.load(f)
        with open(pth + "\\" + "syms.pkl", 'rb') as f:
            self.syms = pkl.load(f)
        with open(pth + "\\" + "word_end_sym.pkl", 'rb') as f:
            self.word_end_sym = pkl.load(f)


if __name__ == '__main__':

    pth = r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\train\train\dialogues_train.txt"

    txt = """"""

    with open(pth, 'r', encoding='utf-8') as f:
        for line in f:
            txt += line

    t = Tokenizer()
    # t.bpe(txt, add_num_per_iter=30, voc_size=5000)

    # print(len(t.tokens))
    # print(len(t.token_to_idx))

    # t.save(r"./tokenizer_param")
    t.load(r"./tokenizer_param")
    txt = """"""
    with open(r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\all.txt", 'r', encoding='utf-8') as f:
        for line in f:
            txt += line
    print(txt)
    enc = t.encode([
        txt
    ])
    print(enc)
    dec = t.decode(enc)
    print(dec)

    # print(len(t.tokens))
    # print(len(set(t.tokens)))
    # print(len(t.token_to_idx))




