# Modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path
from pathlib import Path

import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL

def get_audio_video_data_root():
    data_root = os.environ.get("REST_DATA_ROOT")
    root = Path(data_root or Path(__file__).resolve().parents[2] / "data")
    task_root = root / "Audio_Video_Retrieval"
    return task_root if task_root.exists() else root

def resolve_release_data_path(path, data_root):
    path = Path(path)
    if path.is_absolute():
        return str(path)

    parts = [part for part in path.parts if part not in ("", ".")]
    if parts and parts[0] == "data":
        parts = parts[1:]
    return str(data_root.joinpath(*parts))

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf):
        self.datapath = dataset_json_file
        self.data_root = get_audio_video_data_root()
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        self.mixup = self.audio_conf.get('mixup', 0)
        self.dataset = self.audio_conf.get('dataset')
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        self.noise = self.audio_conf.get('noise', False)
        self.target_length = self.audio_conf.get('target_length')
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))
        self.frame_use = self.audio_conf.get('frame_use', -1)
        self.total_frame = self.audio_conf.get('total_frame', 10)
        # print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))
        self.im_res = self.audio_conf.get('im_res', 224)
        self.num_video_frames = self.audio_conf.get('num_video_frames', 4) # 新增: 每个样本读取几帧

        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])

    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [
                resolve_release_data_path(data_json[i]['wav'], self.data_root),
                data_json[i]['video_id'],
                resolve_release_data_path(data_json[i]['video_path'], self.data_root),
            ]
        data_np = np.array(data_json, dtype=str)
        return data_np

    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['video_id'] = np_data[1]
        datum['video_path'] = np_data[2]
        return datum

    def select_multi_frames(self, video_id, video_path, num_frames=4):
        """采样同一个视频的多帧图片路径"""
        total = self.total_frame
        if total < num_frames:
            frame_indices = [i for i in range(total)]
            frame_indices += [total - 1] * (num_frames - total)
        else:
            frame_indices = np.linspace(0, total - 1, num_frames, dtype=int)
        paths = []
        for frame_idx in frame_indices:
            img_path = f"{video_path}/frame_{frame_idx}/{video_id}.jpg"
            if not os.path.exists(img_path):
                temp_idx = frame_idx
                while not os.path.exists(img_path) and temp_idx > 0:
                    temp_idx -= 1
                    img_path = f"{video_path}/frame_{temp_idx}/{video_id}.jpg"
            paths.append(img_path)
        return paths

    def get_images(self, filenames):
        """读取多帧图片并堆叠为tensor: [num_frames, 3, H, W]"""
        images = []
        for fname in filenames:
            try:
                img = Image.open(fname)
                image_tensor = self.preprocess(img)
            except:
                image_tensor = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            images.append(image_tensor)
        images = torch.stack(images, dim=0)  # [num_frames, 3, H, W]
        return images

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)
            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()
            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]
            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(
                waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')

        target_length = self.target_length
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index):
        datum = self.data[index]
        datum = self.decode_data(datum)

        if random.random() < self.mixup:
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            mix_lambda = np.random.beta(10, 10)
            try:
                fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            # 图像部分这里只做主样本，不做mixup
            try:
                image_paths = self.select_multi_frames(datum['video_id'], datum['video_path'], self.num_video_frames)
                images = self.get_images(image_paths)
            except:
                images = torch.zeros([self.num_video_frames, 3, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading images')
        else:
            try:
                fbank = self._wav2fbank(datum['wav'], None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            try:
                image_paths = self.select_multi_frames(datum['video_id'], datum['video_path'], self.num_video_frames)
                images = self.get_images(image_paths)
            except:
                images = torch.zeros([self.num_video_frames, 3, self.im_res, self.im_res]) + 0.01
                print('there is an error in loading images')

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # images shape is [num_video_frames, 3, H, W], e.g., [4, 3, 224, 224]
        return fbank, images

    def __len__(self):
        return self.num_samples
