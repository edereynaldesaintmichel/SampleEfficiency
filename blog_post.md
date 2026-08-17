It looks like every time I try something, it fails, but I discover something cool:

- Trying to avoid the information collapse during token sampling failed to give any good results, but I discovered the edge bias.
- Trying to predict company growth from SEC 10-K filings failed, but I discovered a way to make transformers better at encoding numerical information. It brought me quite far actually, as a few quant funds were interested.
- Now, trying to implement the "hacking spectral bias" regularization layer failed, but I discovered that shuffling transformer layers during training makes transformers more sample efficient.

## Training a GPT on 2M tokens: how good can it be?

Not good. But I thought it could be a nice thing to check how well a GPT trained on ~1.8M tokens of Shakespeare (the complete works, BPE-tokenized with a 2048-word vocabulary — about 5MB of text) could compress ~90K tokens it had never seen.

The problem here is obviously one of sample efficiency. You can't throw a 10B model at such a task. It'll memorize the training dataset and suck at generalizing. So you need to carefully tune the architecture and the size of the model to maximize sample efficiency on such a small dataset.

All results below are in bits per byte (bpb) on the held-out validation split. Lower is better.

## Baseline: a simple 10M-param transformer

Modern small-model stack (pre-norm RMSNorm, RoPE, QK-norm, ReLU² MLP, tied embeddings), tuned with heavy dropout and weight decay. Best validation: **1.6054 bpb**.

## Implementing the "hacking spectral bias" regularization layer

As a quick reminder, the idea was that deep neural nets tend to struggle to fit high-frequency functions (the classical spectral bias result) because the layered architecture induces a coupling between a deep layer's output frequency and its previous layer's frequency AND amplitude.

Just like the frequency of f(g(x)) depends on both the amplitude and the frequency of g(x).

So the idea was pretty simple: for each layer, sample 10 random directions (Gram-Schmidt orthogonalized), and count the neuron switches happening when sweeping the inputs along these 10 directions (mean_vector + λ·random_direction — accounting for the pre-norms, it's equivalent to rotating the inputs around a certain axis). Use the smoothed PDF of these neuron switches as a proxy for that layer's frequency, in each direction. Then scale the inputs along these directions according to the CDF of the neuron switches, so that each layer's local input amplitude (gradient) matches its local frequency (local neuron-switch count).

Results were disappointing: a minimal positive effect on training loss, but a small negative effect on val loss (**1.6103 bpb**). So, a bad idea all in all.

## Layer looping and shuffling

I then tried looping layers: at each training step, the whole stack is applied k times, with k sampled uniformly in {1, 2, 3, 4}. It helped: **1.5842 bpb** (best at eval time with the stack looped twice, i.e. 16 effective layers).

Then, I asked myself whether shuffling layers would help, on no other basis than "it will surely not help training loss, but I wonder if it'll help generalization". Concretely, each training step runs the blocks in a random "program": a sequence of length 4 to 32, drawn from the model's 8 blocks with replacement, in random order. It does help: **1.5779 bpb** at the same model size.

## All Results:

| Experiment | Params | Best val bpb |
| --- | --- | --- |
| Baseline transformer | 10.5M | 1.6054 |
| + spectral-bias warp layer | 10.5M | 1.6103 |
| Random looping (1–4×) | 10.5M | 1.5842 |
| Random layer programs ("shuffling") | 10.5M | 1.5779 |
| Shuffling, 5M params / 20k steps | 5.3M | 1.5551 |
| + 16-program ensemble | 5.3M | **1.5449** |
| 2-layer looped | 3.1M | 1.6223 |
| Shared wide FFN, param-matched | 5.3M | 1.6200 |

The .md version of this blog post is available here: https://github.com/edereynaldesaintmichel/SampleEfficiency/blob/main/blog_post.md

All the code is available on the repo, so just ask an LLM if you want to go deeper into a certain aspect of these experiments.
