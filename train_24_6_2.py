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

dataset = dataset.MultiDataset([
    dataset.IJCNLPDailyDialog(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\ijcnlp_dailydialog\format.txt",
        tokenizer=tokenizer,
        max_len=config.block_size,
        device=device,
        name="IJCNLPDailyDialog"
    ),
    # dataset.CMUDoG(
    #     tokenizer=tokenizer,
    #     max_len=config.block_size,
    #     device=device
    # ),
    dataset.IJCNLPDailyDialog(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\datasets-CMU_DoG-master\Conversations\format.txt",
        tokenizer=tokenizer,
        max_len=config.block_size,
        device=device,
        name="CMUDoG"
    ),
    dataset.IJCNLPDailyDialog(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\format.txt",
        max_len=config.block_size,
        device=device,
        name="blended_skill_talk"
    ),
    dataset.IJCNLPDailyDialog(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\woz_dialogs\format.txt",
        max_len=config.block_size,
        device=device,
        name="woz_dialogs"
    ),
    dataset.IJCNLPDailyDialog(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\dailydialog\format.txt",
        max_len=config.block_size,
        device=device,
        name="daily dialog"
    ),
    dataset.IJCNLPDailyDialog(
        pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\empatheticdialogues\format.txt",
        max_len=config.block_size,
        device=device,
        name='empathetic dialogues'
    )
])

# dataset = dataset.RandomMultiDataset([{
#     "dataset": dataset.IJCNLPDailyDialog(
#                     tokenizer=tokenizer,
#                     max_len=config.block_size,
#                     device=device
#                 ),
#     "rate": 1.4
#     }, {
#     "dataset": dataset.CMUDoG(
#                     tokenizer=tokenizer,
#                     max_len=config.block_size,
#                     device=device
#                 ),
#     "rate": 1
#     }, {
#     "dataset": dataset.IJCNLPDailyDialog(
#                     pth=r"C:\Users\DELL\Desktop\datasets\supper_replier\blended_skill_talk\all.txt",
#                     max_len=config.block_size,
#                     device=device
#                 ),
#     "rate": 1
#     }]
# )

print("start train")

loader = DataLoader(dataset, batch_size=32, shuffle=True)
gpt = transformer.GPT(config).to(device)

gpt.load_state_dict(torch.load(r"./gpt_param/tst6.pth"))

opt = optim.AdamW(gpt.parameters(), lr=1e-2, weight_decay=1e-3)

total_it = 0

scaler = torch.cuda.amp.GradScaler()

for epoch in range(1):

    for batch_idx, batch in enumerate(loader):

        lr = get_lr(total_it, lr=7e-4, warmup_iters=6000, lr_decay_iters=400000, min_lr=1e-5)
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

        if total_it % 3000 == 0:
            out = gpt.generate(
                idx=tokenizer.encode(["[chat] [user] What about having a meal? [agent]"]),
                max_new_tokens=100, temperature=1.0
            ).cpu().numpy().tolist()
            print(tokenizer.decode(out))
            torch.save(gpt.state_dict(), r"./gpt_param/tst7.pth")






