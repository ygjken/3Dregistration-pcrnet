# for modelnet
python train_pcrnet.py \
    --exp_name iPCRNet_modelnet_emdloss_epoch500 \
    --dataset_path data/modelnet40_ply_hdf5_2048 \
    --epochs 500

python run_pcrnet.py \
    --exp_name iPCRNet_modelnet \
    --pretrained_ptnet checkpoints/iPCRNet_modelnet_emdloss/models/best_ptnet_model.t7 \
    --pretrained_model checkpoints/iPCRNet_modelnet_emdloss/models/best_model.t7

# for dude
python train_pcrnet.py \
    --exp_name iPCRNet_dude \
    --pretrained checkpoints/iPCRNet_modelnet_emdloss/models/best_model.t7 \
    --epochs 200 

