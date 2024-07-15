import torch
import torch.nn as nn
import math
import pickle as pkl
from torchsummary import summary


gpt_pth = r"G:\params\baby1\baby10\300M"
gpt_pth_1B = r"G:\params\baby1\baby10\1B"
gpt_pth_instruct1B = r"G:\params\baby1\baby10\Instruct1B"
gpt_pth_instruct300M = r"G:\params\baby1\baby10\Instruct300M"


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.register_buffer('bias', torch.tril(
            torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)     # b, s, h
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = (y.transpose(1, 2).contiguous().view(B, T, C))
        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FFN(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size + 5, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd)
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, tgt_idx=None):
        device = idx.device

        if idx.dim() == 3:
            idx = idx.squeeze().long()

        _, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            if tgt_idx is None:
                loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
            else:
                b_idx = []
                t_idx = []
                # print(b_idx, t_idx)
                for b, i in enumerate(tgt_idx):
                    t_idx += i
                    b_idx += [b] * len(i)
                loss = torch.nn.CrossEntropyLoss()(logits[b_idx, t_idx, :], targets[b_idx, t_idx])
                # print(b_idx, t_idx)

        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        if isinstance(idx, list):
            idx = torch.tensor(idx, dtype=torch.long, device=self.config.device)
            if len(idx.shape) == 1:
                idx = idx[None, :]

        for _ in range(max_new_tokens):
            idx_cond = (idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:])
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def chat(self, idx, tokenizer, temperature=1.0, max_new_tokens=300, top_k=None, end_sym="[user]", beam_num=None):

        if isinstance(idx, str):
            idx = torch.tensor(tokenizer.encode(idx), dtype=torch.long, device=self.config.device)[None, :]

        end = tokenizer.encode(end_sym)
        start = idx.shape[-1]

        if isinstance(idx, list):
            idx = torch.tensor(idx, dtype=torch.long, device=self.config.device)
            if len(idx.shape) == 1:
                idx = idx[None, :]

        if beam_num is None:
            for _ in range(max_new_tokens):
                idx_cond = (idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:])
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                if idx.shape[-1] > len(end) and idx.squeeze().cpu().numpy().tolist()[-len(end):] == end:
                    break

        else:  # beam search
            scores = [0]
            is_finished = [False]
            idxs = [idx]
            for _ in range(max_new_tokens):
                if False not in is_finished:
                    break
                scores_ = []
                is_finished_ = []
                idxs_ = []
                for i_, idx in enumerate(idxs):
                    idx = (idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:])
                    if is_finished[i_]:
                        scores_.append(scores[i_])
                        is_finished_.append(True)
                        idxs_.append(idx)
                    else:
                        logits, _ = self(idx)
                        logits = logits[:, -1, :] / temperature
                        probs = torch.softmax(logits, dim=-1)  # 1, v
                        _, ids = torch.topk(probs, beam_num)
                        for new_token in ids.squeeze():
                            idxs_.append(
                                torch.cat(
                                    (idx, torch.tensor([[new_token]], dtype=torch.long, device=idx.device)),
                                    dim=1
                                )
                            )
                            is_finished_.append(
                                (new_token == tokenizer.encode('\n')) or
                                (
                                        (idxs_[-1].shape[-1] > len(end)) and
                                        (idxs_[-1].squeeze().cpu().numpy().tolist()[-len(end):] == end)
                                )
                            )
                            scores_.append(scores[i_] + torch.log(probs[0, new_token]))
                _, to_stay = torch.topk(torch.tensor(scores_), beam_num)
                scores = [scores_[i] for i in to_stay.squeeze()]
                idxs = [idxs_[i] for i in to_stay.squeeze()]
                is_finished = [is_finished_[i] for i in to_stay.squeeze()]
            idx = idxs[torch.argmax(torch.tensor(scores))]
        idx = idx.squeeze().cpu().numpy().tolist()
        return tokenizer.decode(idx[start:-len(end)])

    def summary(self, batch_size=1):
        return summary(self, [(self.config.block_size, 1)], batch_size=batch_size)


class Config:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, device):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.device = device

    def __str__(self):
        return f"voc: {self.vocab_size}, max_len: {self.block_size}, " \
               f"layer: {self.n_layer}, head: {self.n_head}, emb: {self.n_embd}"

    def to_dict(self):
        return {
            'vocab_size': self.vocab_size,
            'block_size': self.block_size,
            'n_layer': self.n_layer,
            'n_head': self.n_head,
            'n_embd': self.n_embd,
            'dropout': self.dropout,
            'device': self.device
        }


def load_GPT(dir_pth=r"G:\params\baby1\baby10", config=None):
    config_pth = dir_pth + "\\config.pkl"
    with open(config_pth, "rb") as f:
        old_config = pkl.load(f)

    if config is None:
        config = old_config

    gpt = GPT(config)

    opt = torch.optim.AdamW(gpt.parameters(), lr=1e-2, weight_decay=1e-3)

    if config.block_size == old_config.block_size:
        gpt.load_state_dict(torch.load(dir_pth + "\\param.pth"))
        try:
            opt.load_state_dict(torch.load(dir_pth + "\\opt.pth"))
        except:
            pass
    else:
        # don't load mask
        param_dict = torch.load(dir_pth + "\\param.pth")
        param_dict = {k: v for k, v in param_dict.items() if "attn.bias" not in k}
        params = gpt.state_dict()

        # initialize new pe with the last pe
        wpe = torch.zeros(
            params['transformer.wpe.weight'].shape[0], param_dict['transformer.wpe.weight'].shape[1],
            device=param_dict['transformer.wpe.weight'].device
        )
        wpe[:param_dict['transformer.wpe.weight'].shape[0], :] = param_dict['transformer.wpe.weight']
        wpe[param_dict['transformer.wpe.weight'].shape[0]:, :] = param_dict['transformer.wpe.weight'][-1, :]
        param_dict['transformer.wpe.weight'] = wpe
        params.update(param_dict)

        # load param
        gpt.load_state_dict(params)

    return gpt, opt, config


def save_GPT(gpt, dir_pth=r"G:\params\baby1\baby10", opt=None):
    with open(dir_pth + "\\config.pkl", "wb") as f:
        pkl.dump(gpt.config, f)
    torch.save(gpt.state_dict(), dir_pth + "\\param.pth")
    if opt is not None:
        torch.save(opt.state_dict(), dir_pth + "\\opt.pth")


if __name__ == '__main__':
    conf = Config(
        vocab_size=5500,
        block_size=512,
        n_layer=3,
        n_head=2,
        n_embd=32,
        dropout=0.1,
        device=None
    )

    gpt = GPT(conf)
    gpt.eval()
    ipt = torch.zeros(32, 512).long()
    print(gpt(ipt)[0])

    save_GPT(gpt)
    gpt, opt, conf = load_GPT()
    gpt.eval()

    print(gpt(ipt)[0])
