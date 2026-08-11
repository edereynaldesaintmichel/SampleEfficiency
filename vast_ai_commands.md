git clone https://github.com/edereynaldesaintmichel/SampleEfficiency.git
cd SampleEfficiency
pip install -r requirements.txt
python prepare_data.py --megabytes 10 --vocab_size 256
python train.py --data data/enwik8_10mb_v256 --run_name baseline
python eval.py runs/baseline/best.pt --data data/enwik8_10mb_v256 --stride 128
