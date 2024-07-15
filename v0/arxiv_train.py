import transformer
import dataset
from tokenizer import Tokenizer
import torch
import torch.optim as optim
from schedule import get_lr
from torch.utils.data import DataLoader


tokenizer = Tokenizer()

device = torch.device("cuda")

config = transformer.Config(
    vocab_size=tokenizer.voc_num,
    block_size=384,    # max_len  512
    n_layer=8,
    n_head=6,
    n_embd=6 * 128,
    dropout=0.1,
    device=device
)


dataset = dataset.ArxivTokenDataset(
    tokens_pth=r"D:\txt_datasets\archive (3)\tokens",
    tokenizer=tokenizer,
    max_len=config.block_size,
    device=device
)


print("start train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
gpt = transformer.GPT(config).to(device)

gpt.load_state_dict(torch.load(r"./gpt_param/tst10.pth"))

opt = optim.AdamW(gpt.parameters(), lr=1e-2, weight_decay=1e-3)

total_it = 0

scaler = torch.cuda.amp.GradScaler()

for epoch in range(100):

    for batch_idx, batch in enumerate(loader):

        lr = get_lr(total_it, lr=2e-3, warmup_iters=6000, lr_decay_iters=40000, min_lr=1e-5)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        print(f"new lr: {lr}")
        total_it += 1

        x, y = batch

        with torch.cpu.amp.autocast():
            out, loss = gpt(x, y)

        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # loss.backward()
        # opt.step()

        print(f"epoch {epoch}, batch {batch_idx} / {len(loader)}, loss: {loss}")

    out = gpt.generate(
        idx=tokenizer.encode(['[chat] [user] "A method to generate prompts". [agent]']),
        max_new_tokens=100, temperature=1.0
    ).cpu().numpy().tolist()
    print(tokenizer.decode(out))
    if epoch % 1 == 0:
        gpt.cpu()
        torch.save(gpt.state_dict(), r"./gpt_param/tst10.pth")
        gpt.to(device)





