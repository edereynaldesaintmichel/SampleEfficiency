"""Prepare a dataset: download, split, optionally train a BPE, write token bins.

Datasets:
  enwik8       first N MB, contiguous train/val[/test] split
  shakespeare  Complete Works (Gutenberg pg100), WORK-LEVEL split: whole plays
               held out for val and test, so "generalization" means predicting
               unseen works, not unseen halves of seen works.

Default is byte-level (vocab=256). With --vocab_size > 256, trains a byte-level
BPE on the TRAIN split only (no val/test leakage) and encodes all splits with it.

Outputs into data/<name>/:
  train.bin, val.bin[, test.bin]   token ids (uint8 if vocab<=256 else uint16)
  meta.json                        vocab size, per-split token/byte counts
  tokenizer.json                   only for BPE runs

The official metric is always bits per byte (bpb) over the SAME held-out bytes,
so runs with different tokenizers are directly comparable. Protocol: select
checkpoints/hyperparameters on val; touch test once, at the very end.
"""

import argparse
import json
import os
import re
import urllib.request
import zipfile

import numpy as np

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"

# held-out whole works: one tragedy + one comedy each, mid-sized,
# non-collaborative; histories stay in train (tetralogies share characters
# across plays, holding one out would leak)
VAL_WORKS = ["THE TRAGEDY OF JULIUS CAESAR", "AS YOU LIKE IT"]
TEST_WORKS = ["THE TRAGEDY OF MACBETH", "TWELFTH NIGHT; OR, WHAT YOU WILL"]


def download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading {url} ...")
        urllib.request.urlretrieve(url, path)


def load_enwik8(megabytes: float, val_frac: float, test_frac: float) -> dict[str, bytes]:
    zip_path = os.path.join("data", "cache", "enwik8.zip")
    download(ENWIK8_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        corpus = zf.read("enwik8")[: int(megabytes * 1_000_000)]
    n = len(corpus)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    splits = {"train": corpus[: n - n_val - n_test], "val": corpus[n - n_val - n_test : n - n_test]}
    if n_test:
        splits["test"] = corpus[n - n_test :]
    return splits


def load_shakespeare() -> dict[str, bytes]:
    txt_path = os.path.join("data", "cache", "pg100.txt")
    download(SHAKESPEARE_URL, txt_path)
    text = open(txt_path, encoding="utf-8").read()
    start = text.index("\n", text.index("*** START")) + 1
    body = text[start : text.index("*** END")]

    # contents list: title lines after "Contents", ending where the body
    # repeats the first title (the first work's heading)
    lines = body.split("\n")
    ci = next(i for i, l in enumerate(lines) if l.strip() == "Contents")
    titles: list[str] = []
    j = ci + 1
    while True:
        s = lines[j].strip()
        if s:
            if titles and s == titles[0]:
                break
            titles.append(s)
        j += 1

    # locate each work heading in contents order, starting AFTER the contents
    # block (line j is the first work's heading — the repeat of titles[0])
    starts = []
    pos = sum(len(l) + 1 for l in lines[:j])
    for t in titles:
        m = re.search(r"^\s*" + re.escape(t) + r"\s*$", body[pos:], re.M)
        assert m, f"work heading not found: {t}"
        starts.append(pos + m.start())
        pos = pos + m.end()
    starts.append(len(body))

    for t in VAL_WORKS + TEST_WORKS:
        assert t in titles, f"held-out work not in contents: {t}"

    splits = {"train": b"", "val": b"", "test": b""}
    for i, t in enumerate(titles):
        chunk = body[starts[i] : starts[i + 1]].encode("utf-8")
        key = "val" if t in VAL_WORKS else "test" if t in TEST_WORKS else "train"
        splits[key] += chunk
        print(f"  {key:5s}  {len(chunk):9,}B  {t}")
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="enwik8", choices=["enwik8", "shakespeare"])
    ap.add_argument("--megabytes", type=float, default=10.0, help="enwik8 only: corpus size in MB")
    ap.add_argument("--val_frac", type=float, default=0.1, help="enwik8 only (shakespeare splits by work)")
    ap.add_argument("--test_frac", type=float, default=0.0, help="enwik8 only")
    ap.add_argument("--vocab_size", type=int, default=256, help="256 = raw bytes; >256 trains byte-level BPE on train split")
    ap.add_argument("--name", type=str, default=None, help="output dir name under data/")
    args = ap.parse_args()

    if args.dataset == "enwik8":
        splits = load_enwik8(args.megabytes, args.val_frac, args.test_frac)
        default_name = f"enwik8_{args.megabytes:g}mb_v{args.vocab_size}"
    else:
        splits = load_shakespeare()
        default_name = f"shakespeare_v{args.vocab_size}"

    out_dir = os.path.join("data", args.name or default_name)
    os.makedirs(out_dir, exist_ok=True)

    if args.vocab_size <= 256:
        ids = {k: np.frombuffer(v, dtype=np.uint8) for k, v in splits.items()}
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
        texts = {k: v.decode("utf-8", errors="replace") for k, v in splits.items()}
        tok.train_from_iterator([texts["train"]], trainer=trainer)  # train split ONLY
        tok.save(os.path.join(out_dir, "tokenizer.json"))
        ids = {k: np.array(tok.encode(t).ids, dtype=np.uint16) for k, t in texts.items()}
        vocab_size = tok.get_vocab_size()

    meta = {"dataset": args.dataset, "vocab_size": vocab_size,
            "dtype": "uint8" if vocab_size <= 256 else "uint16"}
    if args.dataset == "shakespeare":
        meta["val_works"], meta["test_works"] = VAL_WORKS, TEST_WORKS
    for k in splits:
        ids[k].tofile(os.path.join(out_dir, f"{k}.bin"))
        meta[f"{k}_tokens"], meta[f"{k}_bytes"] = int(len(ids[k])), len(splits[k])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {out_dir}")
    for k in splits:
        print(f"  {k:5s} {meta[f'{k}_tokens']:>10,} tokens / {meta[f'{k}_bytes']:>10,} bytes")
    print(f"  vocab: {vocab_size}  ({meta['train_bytes']/meta['train_tokens']:.3f} bytes/token)")


if __name__ == "__main__":
    main()
