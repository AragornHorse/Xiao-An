import transformer
import dataset as D
from tokenizer import Tokenizer
import torch
import torch.optim as optim
from schedule import get_lr
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

tokenizer = Tokenizer()

device = torch.device("cuda")

config = transformer.Config(
    vocab_size=tokenizer.voc_num,
    block_size=416,    # max_len  512
    n_layer=10,
    n_head=8,
    n_embd=8 * 480,
    dropout=0.1,
    device=device
)

num_acc = 160
batch_size = 4

max_lr = 2e-5
warm_up_step = 100
lr_decay_steps = 400000
min_lr = 5e-7

betas = (0.93, 0.9996)

weight_decay = 1e-3

last_it = 310001
total_it = 1

torch.manual_seed(12)


dataset = D.RandomMultiDataset([
        {
            'dataset': D.TxtFolderDataset(D.c4_txt, max_len=config.block_size, device=device),     # 741 G
            'rate': 1.
        }, {
            'dataset': D.TxtFolderDataset(D.s2orc_txt, max_len=config.block_size, device=device),    # 577 G
            'rate': 0.55
        }, {
            'dataset': D.TxtFolderDataset(D.python_txt, max_len=config.block_size, device=device),   # 72 G
            'rate': 0.08
        }, {
            'dataset': D.TxtFolderDataset(D.books3_txt, max_len=config.block_size, device=device),    # 144 G
            'rate': 0.45
        }, {
            'dataset': D.TxtFolderDataset(D.qa_txt, max_len=config.block_size, device=device),    # 0.2 G
            'rate': 0.05
        }, {
            'dataset': D.TxtFolderDataset(D.wiki_txt, max_len=config.block_size, device=device),   # 19 G
            'rate': 0.2
        }, {
            'dataset': D.TxtFolderDataset(D.gutenberg_txt, max_len=config.block_size, device=device),     # ~10 G
            'rate': 0.07
        }, {
            'dataset': D.TxtFolderDataset(D.LoC_txt, max_len=config.block_size, device=device),       # 44 G
            'rate': 0.15
        }, {
            'dataset': D.TxtFolderDataset(D.chat_txt, max_len=config.block_size, device=device),   # 0.05 G
            'rate': 0.02
        }, {
            'dataset': D.TxtFolderDataset(D.email_txt, max_len=config.block_size, device=device),   # 0.09 G
            'rate': 0.07
        }
    ])


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# gpt = transformer.GPT(config).to(device)
# opt = optim.AdamW(gpt.parameters(), lr=1e-2, weight_decay=weight_decay, betas=betas)
# gpt.summary(32)

gpt, opt, config = transformer.load_GPT(transformer.gpt_pth_1B, config)
gpt.to(device)
gpt.summary()
print(config)

set_seed(42)
accelerator = Accelerator(mixed_precision='bf16')  # ['no', 'fp8', 'fp16', 'bf16']

# Send everything through `accelerator.prepare`
loader, gpt, opt = accelerator.prepare(
    loader, gpt, opt
)

loss_ = 0


for epoch in range(1):

    for batch in tqdm(loader):

        lr = get_lr(total_it, lr=max_lr, warmup_iters=warm_up_step, lr_decay_iters=lr_decay_steps, min_lr=min_lr)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        total_it += 1

        if total_it < last_it:
            continue

        x, y = batch

        out, loss = gpt(x, y)

        loss = loss * (1. / max([1, num_acc]))

        loss_ += loss.item()

        accelerator.backward(loss)

        if total_it % num_acc == 0:
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            print(f"epoch {epoch}, lr: {lr}, loss: {loss_}")
            loss_ = 0

        if total_it % 5000 == 0:
            out = gpt.generate(
                idx=tokenizer.encode(["I have to say"]),
                max_new_tokens=100, temperature=1.0
            ).cpu().numpy().tolist()
            print(tokenizer.decode(out))
            opt.zero_grad()
            transformer.save_GPT(gpt, transformer.gpt_pth_1B, opt)

        del x, y, loss, out






