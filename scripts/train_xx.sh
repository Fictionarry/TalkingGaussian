dataset=$1
workspace=$2
gpu_id=$3
audio_extractor='deepspeech' # deepspeech, esperanto, hubert

export CUDA_VISIBLE_DEVICES=$gpu_id

python train_mouth.py -s $dataset -m $workspace --audio_extractor $audio_extractor
python train_face.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor
python train_fuse.py -s $dataset -m $workspace --opacity_lr 0.001 --audio_extractor $audio_extractor

# # Parallel. Ensure that you have aleast 2 GPUs, and over N x 64GB memory for about N x 5k frames (IMPORTANT! Otherwise the computer will crash).
# CUDA_VISIBLE_DEVICES=$gpu_id python train_mouth.py -s $dataset -m $workspace --audio_extractor $audio_extractor & 
# CUDA_VISIBLE_DEVICES=$((gpu_id+1)) python train_face.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor
# CUDA_VISIBLE_DEVICES=$gpu_id python train_fuse.py -s $dataset -m $workspace --opacity_lr 0.001 --audio_extractor $audio_extractor

python synthesize_fuse.py -s $dataset -m $workspace --eval --audio_extractor $audio_extractor
python metrics.py $workspace/test/ours_None/renders/out.mp4 $workspace/test/ours_None/gt/out.mp4