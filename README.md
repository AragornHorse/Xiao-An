# Xiao-An
## Chat bot named Xiao An.
![9055fe25f75dbfb94c32cd272fb0122](https://github.com/AragornHorse/Xiao-An/assets/94287335/78d94196-52d8-482d-9a92-33479efdea5b)

## structure
+ A decoder-only transformer with
  + max token: 384
  + layer: 8
  + head number: 6
  + embedding dim: 6 * 128
+ tokenlize with BPE
  + about 5000 tokens

## Training
+ have pretrained on
  + ijcnlp dailydialog
  + CMU_DoG
  + blended skill talk
  + woz dialogs
  + daily dialog
  + empathetic dialogues
  + for one epoch
  + will be pretrained for one more epoch
+ will be fine-tuned soon
