# vision-bert-CCL
## Prepare dataset and models

### Pretrain Dataset:
conceptual caption dataset, you can download the precomputed features through link (the files are being uploaded)

then unzip your downloaded file, obtain the fold cc, and move it to the fold ./dataset

### Finetune Dataset:
MS coco dataset, you can download the precomputed features through this [link](https://www.dropbox.com/s/4g0x49h302xj7j4/coco.tar.gz?dl=0) 

Alternatively, you can obtain it in command line by

wget https://www.dropbox.com/s/4g0x49h302xj7j4/coco.tar.gz?dl=0

then unzip your downloaded file, obtain the fold coco, and move it to the fold ./dataset

### BERT Model
you can download the pretrained bert model through this [link](https://www.dropbox.com/s/oyzjyzc420essow/pytorch_model.bin?dl=0)

then move your downloaded .bin file to ./bert fold

### Pretrained Vision-Bert Model

you can download the pretrained bert model through this [link](https://www.dropbox.com/s/bmejzretyynhtt8/6layer_mlm_pretrain.pth.tar?dl=0)

then move your downloaded .bin file to ./models fold

## Run Scripts


### Pretrain the vision-bert

python train.py --task pretrain --mlm --cm

### Finetune the pretrained model on the retrieval task:

python train.py --task retrieval --resume models/6layer_mlm_pretrain.pth.tar 



## To do

### Ploy Vision-Text Encoder
