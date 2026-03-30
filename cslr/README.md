# CSLR Notes

## Phoenix Keypoints Artifact

Current Phoenix skeleton artifact:

- `datasets/skeletons/phoenix/landmarks.zarr`
- `datasets/skeletons/phoenix/videos.parquet`
- `datasets/skeletons/phoenix/frames.parquet`
- `datasets/skeletons/phoenix/runs.parquet`

Artifact status:

- total samples: `8257`
- splits: `train=7096`, `dev=519`, `test=642`
- `videos.parquet`, `frames.parquet`, and `landmarks.zarr` are consistent

## Stored Coordinates

Phoenix was extracted with `--image-coords`.

This means:

- stored hand landmarks are MediaPipe image landmarks, not world landmarks
- stored pose landmarks are MediaPipe image landmarks, not world landmarks
- `x` and `y` are normalized image coordinates
- `z` is image-space depth, not metric world depth
- `coords_mode=image` for all `8257` samples

For a sample, the main landmark tensors are:

- `raw/left_xyz`: `(T, 21, 3)`
- `raw/right_xyz`: `(T, 21, 3)`
- `raw/pose_xyz`: `(T, 11, 3)`
- `raw/pose_vis`: `(T, 11)`
- `raw/left_score`: `(T,)`
- `raw/right_score`: `(T,)`
- `raw/left_valid`: `(T,)`
- `raw/right_valid`: `(T,)`
- `raw/ts_ms`: `(T,)`

The natural CSLR joint tensor is:

- `X_coords: (T, 53, 3)` from `left + right + pose`

Deployment-parity default:

- train on `raw/*`, not `pp/*`
- use body-centric normalization on top of stored image coordinates

`pp/*` can still be tested as an offline ablation, but it is not the closest match to iOS inference.

## Confidence Signals To Use

Use only confidence-like signals that can be matched on iOS MediaPipe Tasks output.

Recommended training signals:

1. `pose_vis`
- source: `landmarks.zarr -> raw/pose_vis`
- shape: `(T, 11)`
- meaning: per-joint pose visibility

2. `hand_1_present`
- source: `frames.parquet`
- shape: `(T,)`
- meaning: binary left-hand present mask

3. `hand_2_present`
- source: `frames.parquet`
- shape: `(T,)`
- meaning: binary right-hand present mask

4. `hand_1_score`
- source: `frames.parquet`
- shape: `(T,)`
- meaning: hand-level handedness confidence for the left hand

5. `hand_2_score`
- source: `frames.parquet`
- shape: `(T,)`
- meaning: hand-level handedness confidence for the right hand

Do not use the following for deployment-parity training:

- `hand_1_score_gate`
- `hand_2_score_gate`
- `hand_1_tracker_last_score`
- `hand_2_tracker_last_score`
- `hand_*_state`
- `sp_*`
- `occlusion_*`
- `tracker_*`
- `quality_score`
- `pp_*`

Reason:

- these are extractor-internal recovery or postprocessing signals
- they do not exist as native outputs of iOS MediaPipe Tasks

Important Phoenix-specific note:

- this Phoenix run used `score_source=presence`
- therefore `hand_*_score_gate` is effectively not informative for the model

Observed Phoenix confidence summary:

- `hand_1_score` median: `0.979`
- `hand_2_score` median: `0.979`
- `pose_vis` median: `0.998`
- `pose_vis` mean: `0.942`

## Proposed Training Input Contract

Deployment-oriented CSLR input:

- `X_coords: (T, 53, 3)`
- `X_pose_conf: (T, 11)` from `pose_vis`
- `X_hand_mask: (T, 2)` from `hand_1_present`, `hand_2_present`
- `X_hand_score: (T, 2)` from `hand_1_score`, `hand_2_score`

Recommended preprocessing:

1. concatenate `left hand + right hand + selected pose joints`
2. subtract a body anchor, preferably shoulder midpoint or upper-torso anchor
3. divide by torso or shoulder width scale
4. keep `z` as an optional third channel, but do not rely on it as metric depth
5. optionally add motion and bone streams later

## Phoenix Annotation Files

Raw Phoenix annotation files:

- `datasets/phoenix/phoenix14t.pami0.train.annotations_only.gzip`
- `datasets/phoenix/phoenix14t.pami0.dev.annotations_only.gzip`
- `datasets/phoenix/phoenix14t.pami0.test.annotations_only.gzip`

These files are:

- gzip-compressed Python pickle payloads
- each payload is a `list[dict]`

Each raw row contains only four fields:

- `name`
- `signer`
- `gloss`
- `text`

Raw field meanings:

- `name`: sample id in `split/stem` form, for example `train/01April_2010_Thursday_heute-6694`
- `signer`: signer id such as `Signer04`
- `gloss`: gloss token sequence
- `text`: spoken German translation sentence

Prepared annotation files:

- `datasets/phoenix/prepared/phoenix14t.all.tsv`
- `datasets/phoenix/prepared/phoenix14t.train.tsv`
- `datasets/phoenix/prepared/phoenix14t.val.tsv`
- `datasets/phoenix/prepared/phoenix14t.test.tsv`
- matching `*.jsonl`

Prepared TSV adds:

- `attachment_id`
- `split`
- `orig_split`
- `video_relpath`
- `video_path`
- `video_exists`
- `dataset`
- `signer_id`
- `gloss`
- `translation`
- `text`

Join key to skeletons:

- `videos.parquet.sample_id == prepared_tsv.attachment_id`
- rule: `attachment_id = name.replace('/', '__')`

Split note:

- official Phoenix split is `train / dev / test`
- prepared TSV uses `split=train / val / test`
- `orig_split` keeps the official `dev`

## Annotation Statistics

Prepared Phoenix stats:

- total rows: `8257`
- unique signers: `9`
- gloss vocabulary size: `1115`
- translation vocabulary size: `3001`
- median gloss length: `7`
- median translation length: `14`
- max gloss length: `30`
- max translation length: `53`

Signer distribution is imbalanced:

- biggest signers: `Signer01`, `Signer05`, `Signer04`
- smallest signers: `Signer06`, `Signer02`

Use signer information for:

- analysis
- balanced sampling experiments
- robustness diagnostics

Do not depend on `signer_id` as a required model input if inference on iOS will not know the signer.

## What To Use For Best CSLR Accuracy

If the task is true CSLR on Phoenix-2014T:

1. use `gloss` as the primary target
2. use official `train/dev/test` protocol
3. use `dev` as validation and keep `test` untouched until final evaluation
4. join labels to skeletons through `attachment_id/sample_id`
5. train sequence recognition, not sentence-level classification

Why `gloss`:

- gloss is the standard CSLR supervision target on Phoenix
- translation is the target for sign language translation, not recognition
- gloss vocabulary is much smaller and cleaner than spoken-language translation

For maximum deployable accuracy under iOS parity:

- use `raw` skeletons
- use `pose_vis`, hand present masks, and hand handedness scores
- avoid extractor-only recovery fields in model input

For maximum offline benchmark accuracy:

- compare `raw` vs `pp`
- compare coords-only vs coords+confidence
- compare single-stream vs joint/bone/motion multi-stream inputs

## Metrics

For Phoenix CSLR, the standard primary metric is:

- `WER` on gloss sequences

`Accuracy` and `F1` can be logged as auxiliary metrics, but they are not the main benchmark metric for continuous gloss recognition.

Recommended evaluation setup:

1. primary: gloss `WER`
2. optional auxiliary: token-level precision / recall / F1
3. optional auxiliary: exact sentence accuracy

If a model is optimized only for sentence accuracy or simple F1, it may not align with standard Phoenix CSLR reporting.
