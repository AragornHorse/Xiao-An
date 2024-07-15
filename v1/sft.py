import transformer
from tokenizer import Tokenizer
import dataset as D
import sft_dataset
import torch.optim as optim
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
from schedule import get_lr
from tqdm import tqdm

# torch accelerate
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# training parameters
batch_size = 4
batch_acc = 64

beta = (0.9, 0.999)
max_lr = 2e-5
warm_up_step = 100
lr_decay_steps = 2000000
min_lr = 1e-7

total_it = 1
last_it = 1

# prepare model
tokenizer = Tokenizer()
device = torch.device("cuda")

gpt, opt, config = transformer.load_GPT(transformer.gpt_pth_1B, None)
gpt.to(device)
print(config)

# apply lora
para = gpt.named_modules()
peft_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,
    target_modules=[m[0] for m in para if isinstance(m[1], torch.nn.Linear)]
)
del para

del gpt.config
del opt
gpt = get_peft_model(gpt, peft_config)
gpt.print_trainable_parameters()

# sft dataset
dataset = D.RandomMultiDataset([
    {
        'dataset': D.RandomMultiDataset([
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
        ], need_idx=True),
        'rate': 0.25
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.health_care_magic_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.13
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.databricks_dolly_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.01
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.instruct_data_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.09
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.open_hermes_3, config.block_size, tokenizer, device=device
        ),
        'rate': 0.447
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.open_hermes_2, config.block_size, tokenizer, device=device
        ),
        'rate': 0.391
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.open_hermes_1, config.block_size, tokenizer, device=device
        ),
        'rate': 0.315
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.open_hermes_0, config.block_size, tokenizer, device=device
        ),
        'rate': 0.392
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.financial_instruct_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.24
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.auto_cot_train_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.003
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.auto_cot_val_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.0004
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.chatbot_instruction_test, config.block_size, tokenizer, device=device
        ),
        'rate': 0.0249
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.chatbot_instruction_train, config.block_size, tokenizer, device=device
        ),
        'rate': 0.0997
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.cot_reformatted_0, config.block_size, tokenizer, device=device
        ),
        'rate': 0.559 / 2
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.cot_reformatted_1, config.block_size, tokenizer, device=device
        ),
        'rate': 0.614 / 2
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.tofu_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.004
    }, {
        'dataset': sft_dataset.JsonlMenuDataset(
            sft_dataset.open_instruct_pth, config.block_size, tokenizer, device=device
        ),
        'rate': 0.088
    }
], need_idx=True)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sft_dataset.collate_fn_without_pad)

opt = optim.AdamW(gpt.parameters(), lr=1e-5, betas=beta)

# accelerate
accelerator = Accelerator(mixed_precision='bf16')   # ['no', 'fp8', 'fp16', 'bf16']
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

        x, y, idx = batch

        out, loss = gpt(x, y, idx)

        loss = loss * (1. / max([1, batch_acc]))

        loss_ += loss.item()

        accelerator.backward(loss)

        if total_it % batch_acc == 0:
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            print(f"epoch {epoch}, lr: {lr}, loss: {loss_}")
            loss_ = 0

        if total_it % 5000 == 0:
            gpt_base = gpt.merge_and_unload()
            out = gpt_base.generate(
                idx=tokenizer.encode(["[user] Tell me your name. [agent] "]),
                max_new_tokens=100, temperature=1.0
            ).cpu().numpy().tolist()
            print(tokenizer.decode(out))
            opt.zero_grad()
            transformer.save_GPT(gpt_base, transformer.gpt_pth_instruct1B, opt)
            del gpt_base

        del x, y, loss, out
