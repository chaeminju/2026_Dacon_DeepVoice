#!/bin/bash
# 하나의 화자에 대해 RVC 학습 파이프라인을 처음부터 끝까지 실행한다.
# 사용법: rvc_train_speaker.sh <experiment_name> <source_wav_dir> [total_epoch]
set -euo pipefail

EXP_NAME="$1"
SRC_DIR="$2"
TOTAL_EPOCH="${3:-10}"

source /opt/conda/etc/profile.d/conda.sh
conda activate rvc
cd "$(dirname "$0")/../third_party/rvc"

mkdir -p "logs/${EXP_NAME}" assets/weights

echo "[1/6] preprocess"
python -m train.preprocess "$SRC_DIR" 32000 4 "logs/${EXP_NAME}" False 3.7

echo "[2/6] extract_f0"
python -m train.dataset.extract_f0 cuda 1 0 0 "logs/${EXP_NAME}" true

echo "[3/6] extract_hubert_feature"
python -m train.dataset.extract_hubert_feature cuda:0 1 0 0 "logs/${EXP_NAME}" v2 true

echo "[4/6] build filelist + config"
python -c "
import os, shutil, random

exp_dir = 'logs/${EXP_NAME}'
gt_wavs_dir = f'{exp_dir}/0_gt_wavs'
feature_dir = f'{exp_dir}/3_feature768'
f0_dir = f'{exp_dir}/2a_f0'
f0nsf_dir = f'{exp_dir}/2b-f0nsf'

names = (
    set(n.split('.')[0] for n in os.listdir(gt_wavs_dir))
    & set(n.split('.')[0] for n in os.listdir(feature_dir))
    & set(n.split('.')[0] for n in os.listdir(f0_dir))
    & set(n.split('.')[0] for n in os.listdir(f0nsf_dir))
)
print('matched names:', len(names))

now_dir = os.path.abspath('.')
spk_id = 0
opt = []
for name in sorted(names):
    line = f'{now_dir}/{gt_wavs_dir}/{name}.wav|{now_dir}/{feature_dir}/{name}.npy|{now_dir}/{f0_dir}/{name}.wav.npy|{now_dir}/{f0nsf_dir}/{name}.wav.npy|{spk_id}'
    opt.append(line)

for _ in range(2):
    line = f'{now_dir}/logs/mute/0_gt_wavs/mute32k.wav|{now_dir}/logs/mute/3_feature768/mute.npy|{now_dir}/logs/mute/2a_f0/mute.wav.npy|{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy|{spk_id}'
    opt.append(line)

random.shuffle(opt)
with open(f'{exp_dir}/filelist.txt', 'w', encoding='utf8') as f:
    f.write('\n'.join(opt))
print('wrote filelist.txt with', len(opt), 'lines')

shutil.copy('configs/v2/32k.json', f'{exp_dir}/config.json')
"

echo "[5/6] train"
python -m train.train -e "${EXP_NAME}" -sr 32k -f0 1 -bs 8 -g 0 -te "${TOTAL_EPOCH}" -se "${TOTAL_EPOCH}" \
  -pg assets/pretrained_v2/f0G32k.pth -pd assets/pretrained_v2/f0D32k.pth \
  -l 0 -c 1 -sw 1 -v v2

echo "[6/6] train_index"
python -m train.train_index "${EXP_NAME}" v2 "assets/indices" 4 single

echo "DONE: ${EXP_NAME}"
