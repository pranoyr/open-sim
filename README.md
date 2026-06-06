# Open Sim

> [!WARNING]
> **Work In Progress (WIP)**

A Tool to convert human videos doing actions (e.g., cooking) to realistic robotic videos + actions for training/fine-tuning Foundation VLM/VAM models.

## Installation

This project uses `uv` for lightning-fast Python dependency management.

1. Ensure you have `uv` installed (`pip install uv` or via curl).
2. Clone the repository and navigate into the root directory.
3. Install dependencies by running:

```bash
uv sync
```



## Citations

```bibtex
@inproceedings{Pai2025mimicvideoVM,
    title   = {mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs},
    author  = {Jonas Pai and Liam Achenbach and Victoriano Montesinos and Benedek Forrai and Oier Mees and Elvis Nava},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:283920528}
}
```

```bibtex
@misc{kim2026cosmospolicyfinetuningvideo,
    title   = {Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning},
    author  = {Moo Jin Kim and Yihuai Gao and Tsung-Yi Lin and Yen-Chen Lin and Yunhao Ge and Grace Lam and Percy Liang and Shuran Song and Ming-Yu Liu and Chelsea Finn and Jinwei Gu},
    year    = {2026},
    eprint  = {2601.16163},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI},
    url     = {https://arxiv.org/abs/2601.16163},
}
```
