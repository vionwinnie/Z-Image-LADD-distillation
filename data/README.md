# Training Data

## Data Format

Text-only prompts stored as a JSON array:

```json
[
  {
    "text": "A red sports car on a cobblestone street",
    "subject": "S7",
    "style": "T1",
    "camera": "C1",
    "source": "parti_prompts",
    "language": "en"
  }
]
```

The training script only uses the `text` field. Other fields (`subject`, `style`, `camera`, `source`, `language`) are for taxonomy tracking.

## Why Text-Only (No Images)

LADD distillation does not use real images. The teacher model generates synthetic latents from prompts during training. The student learns to match the teacher's distribution through adversarial feedback, not by training on real image-text pairs.

## Data Sources

8 benchmarks downloaded via `prepare_prompts.py`:

| Benchmark | Approx. Count | HuggingFace Dataset | Notes |
|-----------|---------------|---------------------|-------|
| PartiPrompts | ~1,600 | `nateraw/parti-prompts` | |
| GenAI-Bench | ~500 | `TIGER-Lab/GenAI-Bench` | |
| DrawBench | ~200 | `sayakpaul/drawbench` | |
| GenEval | ~550 | `djghosh13/geneval` | |
| DPG-Bench | ~1,065 | `Jialuo21/DPG-Bench` | Dense/long prompts |
| OneIG-ZH | ~1,320 | `OneIG-Bench/OneIG-Bench` | Chinese language prompts |
| CVTG-2K | ~2,000 | `dnkdnk/CVTG-2K` | Text rendering prompts |
| LongText-Bench | ~320 | `X-Omni/LongText-Bench` | Long text rendering |

## Filtering

- Prompts shorter than 5 words (English) or 4 characters (Chinese) are removed.
- Exact case-insensitive deduplication.

## MECE Taxonomy

Each prompt is classified on 3 axes:

- **Subject** (14 categories): People, Animals, Food, etc.
- **Style** (7 categories): Photorealistic, TraditionalArt, etc.
- **Camera** (8 categories): Standard, Macro, WideAngle, etc.

## Directory Structure

```
data/
├── README.md                       # This file
├── prepare_prompts.py              # Downloads benchmarks, classifies, creates splits
├── generate_prompts.py             # Gap-fills to ~10K prompts via Claude API
├── all_classified_prompts.json     # ~7K+ classified benchmark prompts
├── debug/
│   └── metadata.json               # 73 prompts (1 per Subject x Style cell) for smoke testing
└── train/
    └── metadata.json               # Full training set (~10K prompts, requires generate_prompts.py)
```

## How to Use

**Debug (smoke test):**
```bash
--train_data_meta=data/debug/metadata.json
```

**Full training (~10K prompts, requires `ANTHROPIC_API_KEY`):**
```bash
python data/prepare_prompts.py
python data/generate_prompts.py
# then pass --train_data_meta=data/train/metadata.json
```

**Benchmarks only (no API key needed):**
```bash
python data/prepare_prompts.py
# then pass --train_data_meta=data/all_classified_prompts.json
```

## Preprocessing Scripts

| Script | API Key Required | What It Does |
|--------|-----------------|--------------|
| `prepare_prompts.py` | No | Downloads benchmarks from HuggingFace, classifies prompts, creates splits |
| `generate_prompts.py` | Yes (`ANTHROPIC_API_KEY`) | Fills taxonomy gaps with Claude-generated prompts to reach ~10K total |
