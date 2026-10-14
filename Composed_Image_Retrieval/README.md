# Composed image retrieval

This task uses BLIP-2 for FashionIQ (FIQ) and CIRR. REST adapts a composed query made from a reference image and modifying text. FIQ reports Recall@10; CIRR reports Recall@1 with the benchmark evaluator. `-O` contains individual image and text corruptions, while `-D` contains mixed corruptions.

## Data and checkpoints

Download `FIQ-O`, `FIQ-D`, `CIRR-O`, and `CIRR-D` from the [dataset collection](https://drive.google.com/drive/folders/17QyJ9Y52XB67jektNvhUk6jKmIQnoatF?usp=drive_link). Download `blip2.tar.gz`, `sprc_fiq.tar.gz`, and `sprc_cirr.tar.gz` from the [weight collection](https://drive.google.com/drive/folders/1tVuNY9l520BY6SvDdzzmNhF_wCrIaUei?usp=drive_link). Put the archives in their matching folders under `release/release_data/` and `release/weight/`. From `release/`:

```bash
mkdir -p data/Composed_Image_Retrieval weight/Composed_Image_Retrieval
tar -xzf release_data/Composed_Image_Retrieval/FIQ-O/FIQ-O.tar.gz -C data/Composed_Image_Retrieval
tar -xzf release_data/Composed_Image_Retrieval/FIQ-D/FIQ-D.tar.gz -C data/Composed_Image_Retrieval
tar -xzf release_data/Composed_Image_Retrieval/CIRR-O/CIRR-O.tar.gz -C data/Composed_Image_Retrieval
tar -xzf release_data/Composed_Image_Retrieval/CIRR-D/CIRR-D.tar.gz -C data/Composed_Image_Retrieval
tar -xzf weight/Composed_Image_Retrieval/blip2.tar.gz -C weight/Composed_Image_Retrieval
tar -xzf weight/Composed_Image_Retrieval/sprc_fiq.tar.gz -C weight/Composed_Image_Retrieval
tar -xzf weight/Composed_Image_Retrieval/sprc_cirr.tar.gz -C weight/Composed_Image_Retrieval
```

Extract each `-O` dataset before its `-D` partner; the DQS image-split folder links to OQS. The BLIP-2 archive provides `BLIP2/blip2_pretrained.pth` and `BLIP2/eva_vit_g.pth`. The task checkpoints are `SPRC/sprc_fiq.pt` and `SPRC/sprc_cirr.pt`. Set `REST_WEIGHT_ROOT` to the extracted task weight folder so initialization finds both BLIP-2 files.

## Run

Run from `release/code/Composed_Image_Retrieval/` after installing the environment in the [main README](../README.md):

```bash
REST_WEIGHT_ROOT=../../weight/Composed_Image_Retrieval CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset fashioniq --data-root ../../data/Composed_Image_Retrieval \
  --model-path ../../weight/Composed_Image_Retrieval/SPRC/sprc_fiq.pt \
  --image_corruption noise --text_corruption character \
  --method REST --setting OQS --num_workers 2 \
  --feature_save_path /tmp/rest_runs/fiq_features

REST_WEIGHT_ROOT=../../weight/Composed_Image_Retrieval CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset fashioniq --data-root ../../data/Composed_Image_Retrieval \
  --model-path ../../weight/Composed_Image_Retrieval/SPRC/sprc_fiq.pt \
  --image_corruption mixed --text_corruption mixed \
  --method REST --setting DQS --num_workers 2 \
  --feature_save_path /tmp/rest_runs/fiq_d_features
```

For CIRR, use `--dataset cirr --model-path ../../weight/Composed_Image_Retrieval/SPRC/sprc_cirr.pt` and a distinct `--feature_save_path`; also pass `--output-dir /tmp/rest_runs/cirr_predictions`. For the paper's FIQ-to-CIRR and CIRR-to-FIQ tests, evaluate the clean target dataset with the *other* dataset's checkpoint and give each checkpoint a separate feature cache.

The CIRR entrypoint writes test-set ranking JSON files. The public test labels are not in this archive, so CIRR test Recall@1 requires the official benchmark evaluator. The released archives were extracted and their OQS/DQS image, caption, split, and model paths were checked with the loader layout.

For limited host memory, use `--num_workers 2` (or `0`). This controls both gallery feature extraction and query loading. It does not change the batch size or adaptation settings. Keep feature caches separate for different checkpoints.
