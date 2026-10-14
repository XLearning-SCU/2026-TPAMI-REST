# Image–text retrieval

This task uses BLIP for image-to-text (`i2t`, TR@1) and text-to-image (`t2i`, IR@1) retrieval. `configs/OQS/` evaluates individual corruptions and real-world targets; `configs/DQS/` evaluates mixed shifts. Use the checkpoint named by the chosen benchmark config.

## Data and checkpoint

Download archives from the [dataset collection](https://drive.google.com/drive/folders/17QyJ9Y52XB67jektNvhUk6jKmIQnoatF?usp=drive_link) and `blip.tar.gz` from the [weight collection](https://drive.google.com/drive/folders/1tVuNY9l520BY6SvDdzzmNhF_wCrIaUei?usp=drive_link). Put downloaded archives in their matching folders under `release/release_data/` and `release/weight/`. From `release/`, for example:

```bash
mkdir -p data/Image_Text_Retrieval weight/Image_Text_Retrieval
tar -xzf 'release_data/Image_Text_Retrieval/Flickr-C (Flickr-O)/flickr-IP.tar.gz' -C data/Image_Text_Retrieval
tar -xzf 'release_data/Image_Text_Retrieval/Flickr-C (Flickr-O)/flickr-TP.tar.gz' -C data/Image_Text_Retrieval
tar -xzf 'release_data/Image_Text_Retrieval/Flickr-D/Flickr-D.tar.gz' -C data/Image_Text_Retrieval
tar -xzf weight/Image_Text_Retrieval/blip.tar.gz -C weight/Image_Text_Retrieval
```

The three Flickr archives create `flickr-IP/`, `flickr-TP/`, and `Flickr-D/`. `Flickr-D` contains relative links to the first two, so extract it last. COCO uses the analogous `coco-IP`, `coco-TP`, and `COCO-D` packages. Clean Flickr, COCO, and Nocaps, plus Fashion-Gen, RSICD, and RSITMD, are separate dataset archives in the collection. Extract each needed archive directly into `data/Image_Text_Retrieval/`; do not add another folder level.

The BLIP archive creates `weight/Image_Text_Retrieval/BLIP/`. Use `model_base_retrieval_flickr.pth` for Flickr corruption benchmarks, `model_base_retrieval_coco.pth` for COCO corruptions, and `model_base.pth` for the paper's General-to-Flickr/COCO/Nocaps real-world tests. The model path supplied below overrides the YAML checkpoint path without changing its other settings.

## Run

Run from `release/code/Image_Text_Retrieval/` after installing the environment in the [main README](../README.md). The Flickr-O examples reproduce the two query directions:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port=19543 main.py \
  --config configs/OQS/QS/I2T/BLIP_VITB/Flickr/Flickr30K_IP_gaussian_noise_5.yaml \
  --model-path ../../weight/Image_Text_Retrieval/BLIP/model_base_retrieval_flickr.pth \
  --retrieval i2t --method REST --setting OQS --output_dir /tmp/rest_runs/flickr_o_i2t

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master_port=19544 main.py \
  --config configs/OQS/QS/T2I/BLIP_VITB/Flickr/annotation_RandomCharAug_substitute_7.yaml \
  --model-path ../../weight/Image_Text_Retrieval/BLIP/model_base_retrieval_flickr.pth \
  --retrieval t2i --method REST --setting OQS --output_dir /tmp/rest_runs/flickr_o_t2i
```

For Flickr-D, use `configs/DQS/I2T/BLIP_VITB/retrieval_flickr.yaml` or the matching `T2I` YAML and change `--setting DQS`. For COCO-O/D, select the corresponding `COCO/` OQS YAML or `retrieval_coco.yaml` DQS YAML and the COCO checkpoint. The `configs/OQS/QGS/` tree contains the real-world Flickr, COCO, and Nocaps targets; those use the general BLIP checkpoint. Fashion-Gen is a DQS benchmark, while RSICD and RSITMD are additional real-world targets.

The selected Recall@1 appears as `test_txt_r1` for `i2t` and `test_img_r1` for `t2i`. Keep the YAML learning rate, batch size, and corruption level when comparing with the paper. The BLIP ViT-B image-query example above uses two GPUs so its configured global batch size fits in memory.
