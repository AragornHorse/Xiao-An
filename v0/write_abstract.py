import transformer
from tokenizer import Tokenizer
import torch


tokenizer = Tokenizer()

device = torch.device("cpu")

config = transformer.Config(
    vocab_size=tokenizer.voc_num,
    block_size=384,    # max_len  512
    n_layer=8,            # 6
    n_head=6,
    n_embd=6 * 128,
    dropout=0.1,
    device=device
)

gpt = transformer.GPT(config).to(device)

gpt.eval()


gpt.load_state_dict(torch.load(r"./gpt_param/tst10.pth"))


while True:
    title = input("title >>>")
    head = input("the start of the abstract >>>")
    resp = gpt.chat(
        f'\n[chat] [user] "{title}". [agent] {head}',
        tokenizer, end_sym="[user]", max_new_tokens=500, beam_num=None
    )

    if '\n' in resp:
        print(head + resp.split('\n')[0])
        print("\nsection over\n")
    else:
        print(head + resp)


