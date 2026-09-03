git clone https://github.com/edereynaldesaintmichel/SampleEfficiency.git
cd SampleEfficiency
source /venv/main/bin/activate
uv pip install tokenizers

# --- shakespeare (primary: vocab 2048 beat byte-level empirically) ---
python prepare_data.py --dataset shakespeare --vocab_size 2048
python train.py --data data/shakespeare_v2048 --run_name shak_v2048 --steps 60000 --dropout 0.3 --weight_decay 0.25
python eval.py runs/shak_v2048/best.pt --data data/shakespeare_v2048 --stride 128
# final report ONLY (never for selection):
# python eval.py runs/shak_v2048/best.pt --data data/shakespeare_v2048 --stride 128 --split test

# byte-level comparison (bpb is directly comparable across tokenizers):
# python prepare_data.py --dataset shakespeare --vocab_size 256
# python train.py --data data/shakespeare_v256 --run_name shak_v256 --steps 60000 --dropout 0.3 --weight_decay 0.25

# --- generalization probe (10 gens, ~1MB-of-text train subset, 1000 steps each) ---
# shared init (--init_seed); per-gen noise = data order + dropout masks only
# correlations across train/proxy/real-val bpb + pairwise spectral/KL distances + ensemble
git pull
python gen_probe.py --data data/shakespeare_v2048 --run_name gen_probe
# results: runs/gen_probe/results.json (per-seed checkpoints cached, resumable)

# --- enwik8 ---
python prepare_data.py --dataset enwik8 --megabytes 10 --vocab_size 2048
python train.py --data data/enwik8_10mb_v2048 --run_name best_v2048 --steps 100000
python eval.py runs/best_v2048/best.pt --data data/enwik8_10mb_v2048 --stride 128

# --- per-switch amplitude floor (edge bias), 5M/20k shuffle recipe, 2026-09-03 ---
# activations: step relu²-c·H(z) (jump of size c at every switch), dead relu(z-c)², noisy relu(z+c·ε)²
# calibration on shak_shuffle_5M_20k/best.pt: pre-act rms 0.28, positive-z median 0.046, 75% 0.157, 90% 0.288
# baseline is re-run on the same box (cross-machine noise ~0.01 bpb). 5M runs are small: 3 fit concurrently on 16GB.
git pull
mkdir -p runs
printf '%s\n' "relu2 0" "step 0.0025" "step 0.01" "dead 0.05" "noisy 0.1" | xargs -P 3 -L 1 bash -c \
  'python train_shuffle.py --data data/shakespeare_v2048 --run_name shak_5M_act_$0_$1 --n_embd 224 --n_head 4 \
   --steps 20000 --dropout 0.3 --weight_decay 0.25 --grad_accum 2 --act $0 --act_c $1 > runs/shak_5M_act_$0_$1.log 2>&1'
# watch:  tail -n 3 runs/shak_5M_act_*.log ;  grep "new best" runs/shak_5M_act_*.log | tail
# copy off BEFORE stopping the instance:  tar czf act_sweep.tgz runs/shak_5M_act_*  (logs + best.pt)
