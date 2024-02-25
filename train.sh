data_folder="./ImageNet100"
vit_folder=""
vqgan_folder="./pretrained_maskgit/VQGAN/"
writer_log="./logs/"
num_worker=4
bsize=16
cfg_w=0
drop_label=0.0

# Single GPU
python main.py  --bsize ${bsize} --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" --writer-log "${writer_log}" --num_workers ${num_worker} --img-size 256 --cfg_w ${cfg_w} --drop-label ${drop_label}   #--epoch 301 --resume
# Multiple GPUs single node
# torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py  --bsize ${bsize} --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" --writer-log "${writer_log}" --num_workers ${num_worker} --img-size 256 --epoch 301 --resume