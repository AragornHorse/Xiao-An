# Xiao-An

## Who is Xiao An?
+ Xiao An is made by pyhorse, aiming to become a solemate.

## How old is Xiao An?
+ Xiao An is now version 0.0, 0.1

### Structure
+ A decoder-only transformer with
  + max token: 384
  + layer: 8
  + head number: 6
  + embedding dim: 6 * 128
+ tokenlize with BPE
  + about 5000 tokens
+ will try larger size

### Training
+ 0.0 is a chat version
 ![9055fe25f75dbfb94c32cd272fb0122](https://github.com/AragornHorse/Xiao-An/assets/94287335/78d94196-52d8-482d-9a92-33479efdea5b)
  + Have been pretrained on
    + ijcnlp dailydialog
    + CMU_DoG
    + blended skill talk
    + woz dialogs
    + daily dialog
    + empathetic dialogues
    + for one epoch
    + will be pretrained for one more epoch
  + Have been fine-tuned for 2 epochs
+ 0.1 is used to generate abstracts by inputting title
  + trained on Arxiv
  + examples
    + [usr] "KGPI: Kernel Gaussian Process for Image generation".
    + [agent] Optical flow estimation plays a crucial role in many areas of versatility .In this interdisciplinary mind ,a general application of texture ,appearance and nonlinear shadows has been presented as illustrations for tractable generation problems that involve discrete scans of KGPIs .The use of such compromises is an attractive but still difficult task requiring texture and correlations to be applied only for numerous recording volumes millimeters .Many large -scale datasets have been provided separately for objects ,without modifying them so that the model generation procedure is a pivotal option for real -world applications .The usual task for kGPI ,however ,less effectively depends on the scope of a sharp texture estimation process .This paper introduces the KGPI ,an extension -preserving ,combinatorial optimised process regression method that can provide complementary advantages to poisoning learning techniques using two text -text -to -video (T2V )graph neural networks .The key point of this proposed KGPI is the incorporation of Zeno effect in the context of dual robotics ,which has an important motivation to the study of the distribution and relative amount of data explored .Tracking the simplest multi -armed bandit task ,we propose two novel methods to tackle this problem .The first method aims to exploit the structure estimation difficulty in beginning target with the core of truncated relational data ,while the other method enjoys this advantage by utilising shadows only in data preprocessed at runtime .In the second method ,we propose a new data structure that may be used instead of state transparent from the underlying preprocessed datasets .The proposed methods make the supervised

## How long will Xiao An live?
+ Is growing towards version 1
  + become larger
  + more datasets, including c4, books, arxiv will be used for pretraining
  + will be sft for several uses, including chit-chat, writing, etc.
  + may include DPO for RLHF
+ Future
  + version 2
    + including images as a part of input with the help of ResNet, ViT, etc.
    + a new tokenizer will be proposed
  + version 3
    + will try generative reward model
    + more details in sft, including cot, icl

