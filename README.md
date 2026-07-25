# PathFusionRec

Structured semantic–interaction fusion for consumer path inference and generative recommendation.

This repository contains the new project methodology only. The upstream
ActionPiece reproduction remains in `../action_piece/` and is maintained separately.

## Data scope

All new-method experiments use **AmazonReviews2014** only. MovieLens-1M is
retained solely as a completed SASRec reproduction and must not be mixed into
Semantic Only, Interaction Only, or Fusion comparisons. Start with the
processed `Sports_and_Outdoors` split, then use identical preprocessing and
evaluation rules for any additional Amazon categories.

## Bundle semantic encoder

`BundleEncoder` combines two signals rather than averaging contained products:

1. **Global semantics**: title, description, and category embeddings are projected and attention-weighted.
2. **Local structure**: items are scored conditional on the global representation. `item_mask` excludes padding; optional role features, such as quantity or price bucket, can inform scores.
3. **Fusion**: global and local representations are concatenated and projected into the bundle embedding.

The returned `item_attention` is an auditable core-versus-accessory weighting. The model accepts precomputed embeddings, keeping the upstream sentence encoder separate from aggregation.

Run unit tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Real-data smoke experiment

The initial real-data experiment uses raw Amazon `related.bought_together`
relations. AmazonReviews2014 Sports_and_Outdoors is copied to
`data/AmazonReviews2014/Sports_and_Outdoors/` and is ignored by Git because it
is a large source artifact. The experiment trains the encoder to retrieve a
bundle's anchor product from the fused representation, using the cached
sentence-T5 vectors as targets. Run it from this repository with:

```bash
PYTHONPATH=src python3 scripts/run_real_bundle_smoke.py
```

This validates real metadata loading, variable-length bundles, and a learned
attention distribution. It is not a substitute for the later unified
next-item/path-generation benchmark.

## Experiment records

Store each completed experiment in `experiment_results/NN_YYYYMMDD_experiment/`,
where `NN` is an incrementing two-digit sequence number.
Every directory must include raw `metrics.json` and an `analysis.md` using the
template in `experiment_results/analysis_template.md`.
