# ML fundamentals

This repository is my evolving ML fundamentals lab, starting from scalar autograd engines and working through to modern Transformer architectures.
- **Goal**: Build, understand, and document the core components of deep learning models from scratch, while keeping the work reproducible and extensible.
- **Approach**: Implementations in PyTorch and NumPy, detailed walkthroughs, and reflective notes.

| Module | 	Key Topics	| Deliverable |
| ------ | ------------ | ----------- |
| Micrograd | 	Autodiff, backprop, MLP basics	| Scalar engine + training loop |
| Makemore	| Tokenization, embeddings, n-gram models	| Character-level language model |
GPT Mini “Frosty” | 	Transformer blocks, self-attention	| Frost-style text generation model |


---

## Key Result – GPT Mini “Frosty”
A ~7 million-parameter character-level Transformer trained on the poems of Robert Frost.
- **Objective**: Learn Frost’s style and generate plausible new verses.
- **Training**: PyTorch implementation, context size = 256 tokens.
Sample Output:

```
ONENT 


And flower what checks a smile darled out end 
dows for a farmica cellar to fice. 

The fame needn’t know that’s have left. 


Neither expered. His earn one. 


[ 275 ]
```





---

### Deep Dive
Each part walks through the learning process, with explanations, diagrams, and reflections.
- Part 1: micrograd – Scalar autograd, backprop, MLP from scratch.
- Part 2: makemore – Character-level tokenization, embeddings, n-grams.
- Part 3: GPT Mini (Frosty) – Transformer block, attention heads, positional embeddings.

For each part you’ll find:
- Annotated notebooks
- Code comments explaining design decisions
- Reflections on challenges & insights

---
### Future Additions/directions (in here or other repos)
- CNN & RNN implementations for completeness
- More Transformer experiments (e.g., finetuning, interpretability probes)
- Performance benchmarks & visualizations
