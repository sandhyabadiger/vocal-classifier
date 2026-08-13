# vocal-classifier
Classifies speech vs non-speech vocal sounds (laughing, coughing,
crying, breathing, etc). VAD
plus two pretrained sound-tagging models, AST and PANNs

## Setup
```
pip install -r requirements.txt
```

installs things needed for both AST and PANNs

## small test (uses a small sample already included)

```
python eval_ast.py --manifest sample_data/combined_manifest.csv
python eval_panns.py --manifest sample_data/combined_manifest.csv
```

This runs on 300 sample clips already in the repo.
(You should see accuracy around 0.96 for both)

## To run/ evaluate on your own dataset

Put a labeled CSV manifest somewhere (needs columns: filepath,
filename, label, source), then:

```
python eval_ast.py --manifest path/to/your_manifest.csv
python eval_panns.py --manifest path/to/your_manifest.csv
```

Add `--sample_n 300 --balanced` to test on a smaller evenly split
subset instead of the whole file

Each run saves its own results CSV (`eval_ast_results.csv` or
`eval_panns_results.csv`) with per-clip preds

## Checking a single clip

No manifest needed for this, just point at one file or record from
your mic!

```
python inference_ast.py --file some_clip.wav
python inference_panns.py --file some_clip.wav
python inference_ensemble.py --record --duration 3
```

The ensemble script runs both models and combines the answers. If
they agree, it trusts the label. If only one is confident, it goes
with that one. If they disagree, it says "uncertain" instead of
guessing !

## isnide this repo

```
common/        shared code: VAD, audio loading, sound-bucket logic
models/        AST and PANNs wrapper classes
eval_ast.py, eval_panns.py         batch eval scripts, one per model
inference_ast.py, inference_panns.py, inference_ensemble.py   single-clip checks
sample_data/   300 sample clips + manifest, for the quick test above
```