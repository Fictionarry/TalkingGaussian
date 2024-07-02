dataset=$1
workspace=$2
gpu_id=$3

export CUDA_VISIBLE_DEVICES=$gpu_id

python train_mouth.py -s $dataset -m $workspace
python train_face.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005
python train_fuse.py -s $dataset -m $workspace --opacity_lr 0.001

# # Parallel. Ensure that you have aleast 2 GPUs and over 64GB memory.
# CUDA_VISIBLE_DEVICES=$gpu_id python train_mouth.py -s $dataset -m $workspace &
# CUDA_VISIBLE_DEVICES=$((gpu_id+1)) python train_face.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005
# CUDA_VISIBLE_DEVICES=$gpu_id python train_fuse.py -s $dataset -m $workspace --opacity_lr 0.001

python synthesize_fuse.py -s $dataset -m $workspace  --eval
python metrics.py $workspace/test/ours_None/renders/out.mp4 $workspace/test/ours_None/gt/out.mp4