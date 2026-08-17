It looks like every time I try something, it fails, but I discover something cool:

- Trying to avoid the information collapse during token sampling failed to give any good results, but I discovered the edge bias.
- Trying to predict company growth from SEC 10-K filings failed, but I discovered a way to make transformers better at encoding numerical information. It brought me quite far actually, as a few quant funds were interested.
- Now, trying to implement the “hacking spectral bias” regularization layer failed, but I discovered that shuffling transformer layers during training made transformers more sample efficient.

## Training a GPT on 1M tokens: how good can it be?
Not good. But I thought it could be a nice thing to check how good a GPT trained on 900K shakespeare tokens would be at compressing 100K tokens it had never seen.

The problem here is obviously one of sample efficiency. You can't throw a 10B model at such a task. It'll memorize the training dataset and suck at generalizing. So you need to carefully tune the architecture and the size of the model to maximize sample efficiency on such a small dataset.

## Baseline: A simple 10M params transformer.

Got 1.6054 bpb on the validation dataset at best.

## Implementing the "hacking spectral bias" regularization layer.

As a quick reminder, the idea was that Deep Neural Nets tend to struggle fitting high-frequency functions (the classical spectral bias result) because the layered architecture induces a coupling between a deep layer's output frequency and its previous layer's frequency AND amplitude.

Just like the frequency of f(g(x)) depends both on the amplitude and frequency of g(x).

So the idea was pretty simple: for each layer, sample 10 random directions (and gram-schmidt orthogonalize them), and check the number of neuron switches happening when sweeping the inputs in these 10 directions (mean_vector + lambda * random_direction_vector. Accounting for the prenorms, it's equivalent to rotating the inputs around a certain axis). Use the smoothed PDF of these neuron switches as a proxy for that layer's frequency, for each direction. Then, scale the inputs in these directions according to the CDF of these neuron switches, so as to encourage these high-frequency regions.
It's really about making sure that each layer's local input amplitude (gradient) matches its local frequency (local neuron switch count).

Results were disappointing: a minimal positive effect on training loss, but a small negative effect on val loss (1.6103). So, a bad idea all in all.

## Layer looping and shuffling
I then tried looping layers: all layers looped up to 4 times. It helped: {Claude code: val loss}.

Then, I asked myself whether just shuffling layers would help, on no other basis than "it will surely not help training loss, but I wonder if it'll help generalization". It does help. {Claude Code: val loss}


## Going a little further and heavily sharing weights

.md version of this blog post is available here: https://github.com/edereynaldesaintmichel/SampleEfficiency/blob/main/blog_post.md

All the code is available on the repo, so just ask an LLM if you want to go deeper into a certain aspect of these experiments.
