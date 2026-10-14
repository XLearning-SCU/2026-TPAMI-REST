# Video–audio retrieval

This task uses CAV-MAE for video-to-audio (`v2a`, AR@1) and audio-to-video (`a2v`, VR@1) retrieval. AudioSet-O contains individual query corruptions; AudioSet-D contains mixed corruptions. VGGSound is the target for the paper's AudioSet-to-VGGSound test.

## Data and checkpoint

Download `AudioSet-O`, `AudioSet-D`, and, for the cross-dataset test, `VGGSound` from the [dataset collection](https://drive.google.com/drive/folders/17QyJ9Y52XB67jektNvhUk6jKmIQnoatF?usp=drive_link). Download `cav-mae.tar.gz` from the [weight collection](https://drive.google.com/drive/folders/1tVuNY9l520BY6SvDdzzmNhF_wCrIaUei?usp=drive_link). Put the archives in their matching folders under `release/release_data/` and `release/weight/`. From `release/`:

```bash
mkdir -p data/Audio_Video_Retrieval weight/Audio_Video_Retrieval
tar -xzf release_data/Audio_Video_Retrieval/AudioSet-O/AudioSet-O.tar.gz -C data/Audio_Video_Retrieval
tar -xzf release_data/Audio_Video_Retrieval/AudioSet-D/AudioSet-D.tar.gz -C data/Audio_Video_Retrieval
tar -xzf release_data/Audio_Video_Retrieval/VGGSound/VGGSound.tar.gz -C data/Audio_Video_Retrieval
tar -xzf weight/Audio_Video_Retrieval/cav-mae.tar.gz -C weight/Audio_Video_Retrieval
```

Extract `AudioSet-O` before `AudioSet-D`: the DQS package links its clean audio and video folders to OQS. The archive roots are `AudioSet-O/`, `AudioSet-D/`, `VGGSound/`, and `cav-mae/`. The checkpoint is `weight/Audio_Video_Retrieval/cav-mae/audio_model.21.pth`. The JSON files under `config/` already point to these data folders; `REST_DATA_ROOT` can override the shared data root.

## Run

Run from `release/code/Audio_Video_Retrieval/` after installing the environment in the [main README](../README.md):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --json_root config/OQS/v2a/video_noise_blur_digital.json \
  --model_path ../../weight/Audio_Video_Retrieval/cav-mae/audio_model.21.pth \
  --retrieval v2a --method REST --setting OQS

CUDA_VISIBLE_DEVICES=0 python main.py \
  --json_root config/DQS/v2a/video_mixed.json \
  --model_path ../../weight/Audio_Video_Retrieval/cav-mae/audio_model.21.pth \
  --retrieval v2a --method REST --setting DQS
```

Use `config/OQS/a2v/audio_noise_weather.json --retrieval a2v` for an OQS audio query, or `config/DQS/a2v/audio_mixed.json --retrieval a2v --setting DQS` for DQS. For AudioSet-to-VGGSound, use `config/OQS/vgg_3k.json --setting OQS` and run each retrieval direction separately with the same AudioSet checkpoint. The printed Recall@1 is AR@1 for `v2a` and VR@1 for `a2v`.

The released manifests and the extracted archives were checked together: all eight evaluation manifests resolve to audio files and video folders after extraction. Keep the config, checkpoint, query direction, and corruption aligned with the paper table you compare against.
