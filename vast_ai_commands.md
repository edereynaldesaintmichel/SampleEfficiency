git clone https://github.com/edereynaldesaintmichel/SampleEfficiency.git
cd SampleEfficiency
source /venv/main/bin/activate
uv pip install tokenizers

# --- shakespeare (primary, byte-level: rhyme/meter live at char level, and a
# train-split BPE would tokenize unseen plays worse than seen ones, polluting
# the generalization signal) ---
python prepare_data.py --dataset shakespeare --vocab_size 256
python train.py --data data/shakespeare_v256 --run_name shak_v256 --steps 60000 --dropout 0.3 --weight_decay 0.25
python eval.py runs/shak_v256/best.pt --data data/shakespeare_v256 --stride 128
# final report ONLY (never for selection):
# python eval.py runs/shak_v256/best.pt --data data/shakespeare_v256 --stride 128 --split test

# BPE comparison run, if wanted (bpb is directly comparable across tokenizers):
# python prepare_data.py --dataset shakespeare --vocab_size 2048
# python train.py --data data/shakespeare_v2048 --run_name shak_v2048 --steps 60000 --dropout 0.3 --weight_decay 0.25

# --- enwik8 ---
python prepare_data.py --dataset enwik8 --megabytes 10 --vocab_size 2048
python train.py --data data/enwik8_10mb_v2048 --run_name best_v2048 --steps 100000
python eval.py runs/best_v2048/best.pt --data data/enwik8_10mb_v2048 --stride 128
