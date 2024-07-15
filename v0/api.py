import transformer
from tokenizer import Tokenizer
import torch
import sys

device = torch.device("cpu")

tokenizer = Tokenizer()

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

gpt.load_state_dict(torch.load(r"./gpt_param/tst5.pth"))


ids = sys.argv[1]
resp = gpt.chat(f"\n[chat] [user] {ids} [agent]", tokenizer, end_sym="[user]", max_new_tokens=50)

print(resp)




