import transformer
import dataset as D
from tokenizer import Tokenizer
import torch
import torch.optim as optim
from schedule import get_lr
from torch.utils.data import DataLoader
from tqdm import tqdm


tokenizer = Tokenizer()

device = torch.device("cuda")

config = transformer.Config(
    vocab_size=tokenizer.voc_num,
    block_size=448,    # max_len  512
    n_layer=10,
    n_head=8,
    n_embd=8 * 128,
    dropout=0.1,
    device=device
)

dataset = D.RandomMultiDataset([
        {
            'dataset': D.TxtFolderDataset(D.c4_txt, max_len=config.block_size, device=device),     # 741 G
            'rate': 1.
        }, {
            'dataset': D.TxtFolderDataset(D.s2orc_txt, max_len=config.block_size, device=device),    # 577 G
            'rate': 0.5
        }, {
            'dataset': D.TxtFolderDataset(D.python_txt, max_len=config.block_size, device=device),   # 72 G
            'rate': 0.05
        }, {
            'dataset': D.TxtFolderDataset(D.books3_txt, max_len=config.block_size, device=device),    # 144 G
            'rate': 0.3
        }, {
            'dataset': D.TxtFolderDataset(D.qa_txt, max_len=config.block_size, device=device),    # 0.2 G
            'rate': 0.05
        }, {
            'dataset': D.TxtFolderDataset(D.wiki_txt, max_len=config.block_size, device=device),   # 19 G
            'rate': 0.1
        }, {
            'dataset': D.TxtFolderDataset(D.gutenberg_txt, max_len=config.block_size, device=device),     # ~10 G
            'rate': 0.08
        }, {
            'dataset': D.TxtFolderDataset(D.LoC_txt, max_len=config.block_size, device=device),       # 44 G
            'rate': 0.15
        }, {
            'dataset': D.TxtFolderDataset(D.chat_txt, max_len=config.block_size, device=device),   # 0.05 G
            'rate': 0.05
        }
    ])


print("start train")


loader = DataLoader(dataset, batch_size=32, shuffle=True)
gpt = transformer.GPT(config).to(device)
# gpt, config = transformer.load_GPT(transformer.gpt_pth)

gpt.summary(32)

opt = optim.AdamW(gpt.parameters(), lr=1e-2, weight_decay=1e-3)

total_it = 0

scaler = torch.cuda.amp.GradScaler()


for epoch in range(1):

    for batch in tqdm(loader):

        lr = get_lr(total_it, lr=7e-4, warmup_iters=6000, lr_decay_iters=400000, min_lr=1e-5)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        total_it += 1

        x, y = batch

        with torch.cpu.amp.autocast():
            out, loss = gpt(x, y)

        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        print(f"epoch {epoch}, lr: {lr}, loss: {loss}")

        if total_it % 3000 == 0:
            out = gpt.generate(
                idx=tokenizer.encode(["Hello ?"]),
                max_new_tokens=100, temperature=1.0
            ).cpu().numpy().tolist()
            print(tokenizer.decode(out))
            transformer.save_GPT(gpt, transformer.gpt_pth)






