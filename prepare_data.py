"""Prepare the dataset: download enwik8, take the first N MB, contiguous train/val split.

Default is byte-level (vocab=256). With --vocab_size > 256, trains a byte-level BPE
on the TRAIN split only (no val leakage) and encodes both splits with it.

Outputs into data/<name>/:
  train.bin, val.bin   token ids (uint8 if vocab<=256 else uint16)
  meta.json            vocab size, token/byte counts (needed for bpb)
  tokenizer.json       only for BPE runs

The official metric is always bits per byte (bpb) over the SAME val bytes,
so runs with different tokenizers are directly comparable.
"""

import argparse
import json
import os
import urllib.request
import zipfile

import numpy as np

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"


def download_enwik8(cache_dir: str) -> bytes:
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "enwik8.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {ENWIK8_URL} ...")
        urllib.request.urlretrieve(ENWIK8_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read("enwik8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--megabytes", type=float, default=10.0, help="total corpus size in MB (train+val)")
    ap.add_argument("--val_frac", type=float, default=0.1, help="contiguous tail fraction held out for val")
    ap.add_argument("--vocab_size", type=int, default=256, help="256 = raw bytes; >256 trains byte-level BPE on train split")
    ap.add_argument("--name", type=str, default=None, help="output dir name under data/ (default: enwik8_<MB>mb_v<vocab>)")
    args = ap.parse_args()

    raw = download_enwik8(os.path.join("data", "cache"))
    n_bytes = int(args.megabytes * 1_000_000)
    corpus = raw[:n_bytes]
    split = int(len(corpus) * (1 - args.val_frac))
    train_bytes, val_bytes = corpus[:split], corpus[split:]

    name = args.name or f"enwik8_{args.megabytes:g}mb_v{args.vocab_size}"
    out_dir = os.path.join("data", name)
    os.makedirs(out_dir, exist_ok=True)

    if args.vocab_size <= 256:
        train_ids = np.frombuffer(train_bytes, dtype=np.uint8)
        val_ids = np.frombuffer(val_bytes, dtype=np.uint8)
        vocab_size = 256
    else:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[],
            show_progress=True,
        )
        # train on the train split ONLY
        train_text = train_bytes.decode("utf-8", errors="replace")
        val_text = val_bytes.decode("utf-8", errors="replace")
        tok.train_from_iterator([train_text], trainer=trainer)
        tok.save(os.path.join(out_dir, "tokenizer.json"))
        train_ids = np.array(tok.encode(train_text).ids, dtype=np.uint16)
        val_ids = np.array(tok.encode(val_text).ids, dtype=np.uint16)
        vocab_size = tok.get_vocab_size()

    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    meta = {
        "dataset": "enwik8",
        "vocab_size": vocab_size,
        "dtype": "uint8" if vocab_size <= 256 else "uint16",
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "train_bytes": len(train_bytes),
        "val_bytes": len(val_bytes),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out_dir}")
    print(f"  train: {meta['train_tokens']:,} tokens / {meta['train_bytes']:,} bytes")
    print(f"  val:   {meta['val_tokens']:,} tokens / {meta['val_bytes']:,} bytes")
    print(f"  vocab: {vocab_size}  ({meta['train_bytes']/meta['train_tokens']:.3f} bytes/token)")


if __name__ == "__main__":
    main()
