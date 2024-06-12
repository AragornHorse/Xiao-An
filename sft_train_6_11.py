import transformer
import sft_dataset as S
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


# dataset = dataset.IJCNLPDailyDialog(
#     tokenizer=tokenizer,
#     max_len=config.block_size,
#     device=device
# )

# dataset = dataset.ArxivAbstract(
#     tokenizer=tokenizer,
#     max_len=config.block_size,
#     device=device
# )

dataset = S.MultiDataset([
    S.ChatDataset(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\format.txt",
        tokenizer=tokenizer,
        max_len=config.block_size,
        device=device,
        pretrain_rate=0.02,
        name="IJCNLPDailyDialog"
    ),
    S.ChatDataset(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\format.txt",
        tokenizer=tokenizer,
        max_len=config.block_size,
        device=device,
        pretrain_rate=0.02,
        name="CMUDoG"
    ),
    S.ChatDataset(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\format.txt",
        max_len=config.block_size,
        device=device,
        pretrain_rate=0.02,
        name="blended_skill_talk"
    ),
    S.ChatDataset(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\format.txt",
        max_len=config.block_size,
        device=device,
        pretrain_rate=0.02,
        name="woz_dialogs"
    ),
    S.ChatDataset(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\format.txt",
        max_len=config.block_size,
        device=device,
        pretrain_rate=0.02,
        name="daily dialog"
    ),
    S.ChatDataset(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\format.txt",
        max_len=config.block_size,
        device=device,
        pretrain_rate=0.02,
        name='empathetic dialogues'
    )
])


print("start train")

loader = DataLoader(dataset, batch_size=48, shuffle=True, collate_fn=S.collate_fn)
gpt = transformer.GPT(config).to(device)

gpt.load_state_dict(torch.load(r"./gpt_param/tst9.pth"))

opt = optim.AdamW(gpt.parameters(), lr=1e-2, weight_decay=1e-3)

total_it = 0

scaler = torch.cuda.amp.GradScaler()

for epoch in range(2):

    for batch_idx, batch in enumerate(loader):

        lr = get_lr(total_it, lr=7e-4, warmup_iters=1200, lr_decay_iters=5000, min_lr=1e-5)
        for param_group in opt.param_groups:
            param_group["lr"] = lr
        print(f"new lr: {lr}")
        total_it += 1

        x, y, idx = batch

        with torch.cpu.amp.autocast():
            out, loss = gpt(x, y, idx)

        torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # loss.backward()
        # opt.step()

        print(f"epoch {epoch}, batch {batch_idx} / {len(loader)}, loss: {loss}")

        if total_it % 1000 == 0:
            out = gpt.generate(
                idx=tokenizer.encode(["[chat] [user] What about having a meal? [agent]"]),
                max_new_tokens=100, temperature=1.0
            ).cpu().numpy().tolist()
            print(tokenizer.decode(out))
            torch.save(gpt.state_dict(), r"./gpt_param/tst8.pth")

torch.save(gpt.state_dict(), r"./gpt_param/tst9.pth")






