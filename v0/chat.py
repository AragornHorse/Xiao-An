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

gpt.load_state_dict(torch.load(r"./gpt_param/tst8.pth"))

history = ""

while True:
    ids = input(">>>")
    if len(history) == 0:
        resp = gpt.chat(f"\n[chat] [user] {ids} [agent]", tokenizer, end_sym="[user]", max_new_tokens=50, beam_num=None)
        history += f"\n[chat] [user] {ids} [agent]" + resp
    else:
        resp = gpt.chat(f"{history} [user] {ids} [agent]", tokenizer, end_sym="[user]", beam_num=None)
        history += f"[user] {ids} [agent]" + resp
    # if len(history) > 384:
    #     history = history[-384:]

    if '\n' in resp:
        print(resp.split('\n')[0])
        print("\nsection over\n")
        history = ""
    else:
        print(resp)

# while True:
#     ids = input(">>>")
#     if len(history) == 0:
#         resp = gpt.chat(f"\n {ids} __eou__", tokenizer, end_sym="__eou__")
#         history += f"\n {ids} __eou__" + resp
#     else:
#         resp = gpt.chat(f"{history} __eou__ {ids} __eou__", tokenizer, end_sym="__eou__")
#         history += f"__eou__ {ids} __eou__" + resp
#     if len(history) > 512:
#         history = history[-512:]
#
#     if '\n' in resp:
#         print("\nsection over\n")
#         history = ""
#     else:
#         print(resp)
