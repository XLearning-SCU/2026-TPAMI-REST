# REST: Toward Robust and Harmonious Adaptation for Cross-modal Retrieval

Official code for **Toward Robust and Harmonious Adaptation for Cross-modal Retrieval** by Haobin Li, Zhuqi Liu, Mouxing Yang, and Xi Peng. REST continues our [ICLR 2025 TCR release](https://hbinli.github.io/TCR/). [[Paper]](https://doi.org/10.1109/TPAMI.2026.3729221)

## Query shift

REST studies online query streams whose distribution changes over time. **Online Query Shift (OQS)** evaluates individual shifts; **Diverse Query Shift (DQS)** mixes shifts in one stream. The paper extends image–text retrieval to video–audio and composed image retrieval.

<img src="figs/figure1.png" alt="Figure 1: online and diverse query shift" width="950">

## Method

REST refines retrieval predictions, then adapts the query encoder with query uniformity, query–gallery gap, and query–gallery consistency losses. DQS adds gradient decoupling to preserve general retrieval knowledge.

The release keeps each original implementation's adaptation arithmetic and task defaults. In particular, CAV-MAE and SPRC use `1/(1+KL)` for the DQS gradient weight, while BLIP uses `exp(-KL)`; the paper writes the latter form in its summary.

<img src="figs/method.png" alt="Figure 2: REST method" width="950">

## Benchmarks and code

| Task | Backbone | Datasets | Metric | Instructions |
| --- | --- | --- | --- | --- |
| Image–text retrieval | BLIP | Flickr, COCO, Nocaps, Fashion-Gen, RSICD, RSITMD | TR@1 / IR@1 | [Image_Text_Retrieval](Image_Text_Retrieval/README.md) |
| Video–audio retrieval | CAV-MAE | AudioSet-O/D, VGGSound | AR@1 / VR@1 | [Audio_Video_Retrieval](Audio_Video_Retrieval/README.md) |
| Composed image retrieval | BLIP-2 | FIQ-O/D, CIRR-O/D | CIR@10 / CIR@1 | [Composed_Image_Retrieval](Composed_Image_Retrieval/README.md) |

`-O` and `-D` name the OQS and DQS benchmarks. The earlier TCR collection calls the individual Flickr and COCO corruptions `Flickr-C` and `COCO-C`; those are the Flickr-O and COCO-O data used here.

## Download

- [Datasets on Google Drive](https://drive.google.com/drive/folders/17QyJ9Y52XB67jektNvhUk6jKmIQnoatF?usp=drive_link)
- [Model weights on Google Drive](https://drive.google.com/drive/folders/1tVuNY9l520BY6SvDdzzmNhF_wCrIaUei?usp=drive_link)

Both collections have `Image_Text_Retrieval`, `Audio_Video_Retrieval`, and `Composed_Image_Retrieval` folders. Download the archives required by your task. Extract each dataset archive into the matching folder under `release/data/` and each model archive into the matching folder under `release/weight/`. Every archive includes its own top-level dataset or model directory. Extract an `-O` dataset before its `-D` partner because the `-D` package links shared clean files from `-O`. Each task README lists the extraction paths and checkpoints.

## Environment

The code was checked in `/opt/conda/envs/tta` with Python 3.9.16, PyTorch 2.1.0, CUDA 12.1, and GPU FAISS 1.7.2. For a new environment:

```bash
conda create -n rest python=3.9.16 pip -y
conda activate rest
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Run the commands in the task README from that task's directory. Put metrics, feature caches, and predictions outside `code/`, for example under `/tmp/rest_runs`. `REST_DATA_ROOT` can point to another data root with the same three task folders.

BLIP and BLIP-2 use the Hugging Face `bert-base-uncased` tokenizer. The first run needs network access to download it; if it is already cached and the network is unavailable, prefix the command with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## Reproduce

Each task README gives the required archives, checkpoint, and runnable commands for OQS, DQS, and cross-dataset evaluation:

- [Image–text: Flickr/COCO and other targets](Image_Text_Retrieval/README.md)
- [Video–audio: AudioSet and VGGSound](Audio_Video_Retrieval/README.md)
- [Composed image: FashionIQ and CIRR](Composed_Image_Retrieval/README.md)

The paper contains the full benchmark protocol and reported results. The source-code adaptation settings and task checkpoints are kept in the released configs and entrypoints.

## Citation

```bibtex
@article{li2026toward,
  title={Toward robust and harmonious adaptation for cross-modal retrieval},
  author={Li, Haobin and Liu, Zhuqi and Yang, Mouxing and Peng, Xi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2026},
  publisher={IEEE},
  doi={10.1109/TPAMI.2026.3729221}
}
```

The authors' code is under [Apache 2.0](LICENSE). Bundled LAVIS and CAV-MAE components retain their upstream BSD licenses in the corresponding task folders.
