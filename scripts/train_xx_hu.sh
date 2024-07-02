dataset=$1
workspace=$2

python train_mouth.py -s $dataset -m $workspace --audio_extractor hubert
python train_face.py -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor hubert
python train_fuse.py -s $dataset -m $workspace --opacity_lr 0.001 --audio_extractor hubert
python synthesize_fuse.py -s $dataset -m $workspace --eval --audio_extractor hubert