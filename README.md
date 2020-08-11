# vision-bert-CCL
## Prepare dataset and models

### Pretrain Dataset:
conceptual caption dataset, you can download the precomputed features through this link

then unzip your downloaded file, obtain the fold cc, and move it to the fold ./dataset

### Finetune Dataset:
MS coco dataset, you can download the precomputed features through this link

then unzip your downloaded file, obtain the fold coco, and move it to the fold ./dataset

### BERT Model
you can download the pretrained bert model through this link

then move your downloaded .bin file to ./bert fold

### Pretrained Vision-Bert Model

you can download the pretrained bert model through this link

then move your downloaded .bin file to ./models fold

## Run Scripts

Finetune the pretrained model on the retrieval task:

python train.py --task retrieval --resume models/6layer_mlm_pretrain.pth.tar 




## To do

# Ploy Vision-Text Encoder
