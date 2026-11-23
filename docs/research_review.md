# Prior-art review (Stage A)

Starting bibliography — every entry needs verification of metadata, version, mechanism and code.
Record findings in [`novelty_matrix.csv`](novelty_matrix.csv).

| Work | Relevance to PLE |
|---|---|
| [SNLP: Layer-Parallel Inference via Structured Newton Corrections (2026)](https://arxiv.org/abs/2605.17842) | Closest prior art; check whether speculate-and-reconcile is subsumed |
| [EAGLE (ICML 2024)](https://arxiv.org/abs/2401.15077) | Feature-level prediction, but for future tokens |
| [LayerSkip (ACL 2024)](https://aclanthology.org/2024.acl-long.681/) | Early-layer drafting with self-speculative verification |
| [CQIL (ACL 2024)](https://aclanthology.org/2024.acl-long.394/) | Concurrent quasi-independent layers from approximate inputs |
| [CLaSp (2025)](https://arxiv.org/abs/2505.24196) | In-context layer skip for self-speculation |
| [Parallel Track Transformers (Apple)](https://machinelearning.apple.com/research/parallel-track) | Architectural reduction of sequential dependencies |
| [Leveraging the true depth of LLMs (2025)](https://arxiv.org/abs/2502.02790) | Layer-parallel baseline |
| [ParaFormer (2025)](https://arxiv.org/abs/2510.15425) | Architectural progressive approximation |
| [Speculative Decoding (Leviathan et al.)](https://proceedings.mlr.press/v202/leviathan23a.html), [survey](https://arxiv.org/abs/2502.19732) | Terminology and experimental discipline |

Also search the exact phrase "Predictive Layer Execution" and the acronym PLE across arXiv,
Semantic Scholar, OpenReview, ACM, IEEE, patents and GitHub, and document collisions.

## Novelty boundary

_To be written before implementation starts._
