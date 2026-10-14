import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image
import numpy as np
from data.utils import pre_caption

class rsitmd_retrieval_eval_image(Dataset):
    def __init__(self, transform, img_dir, json_dir, split, max_words=30):
        '''
        img_dir (string): Root directory of images
        json_dir (string): Path to the JSON file
        split (string): Dataset split (e.g. test)
        max_words (int): Maximum number of words in a caption
        '''
        self.transform = transform
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.split = split
        self.max_words = max_words

        self.image = []
        self.text = []
        self.txt2img = {}
        self.img2txt = {}
        self.max_words=max_words
        self._load_annotations()
        
        # print(len(self.image), len(self.text))
        # exit(-1)


    def _load_annotations(self):
        datasets = json.load(open(self.json_dir, "r"))
        txt_id = 0
        img_id_counter = 0
        for image in datasets['images']:
            if img_id_counter >= 450:
                break
            if image['split'] == self.split:
                self.image.append(image['filename'])
                self.img2txt[img_id_counter] = []
                for text in image['sentences']:
                    # caption = self._pre_caption(text['raw'])
                    caption = pre_caption(text['raw'],self.max_words)
                    self.text.append(caption)
                    self.img2txt[img_id_counter].append(txt_id)
                    # self.txt2img[txt_id] = img_id_counter
                    self.txt2img[txt_id] = []
                    self.txt2img[txt_id].append(img_id_counter)
                    txt_id += 1
                img_id_counter += 1

    def _pre_caption(self, caption):
        words = caption.split()
        if len(words) > self.max_words:
            words = words[:self.max_words]
        return ' '.join(words).capitalize()

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        if index not in self.img2txt:
            raise KeyError(f"Index {index} not found in img2txt")
        
        image_path = os.path.join(self.img_dir, self.image[index])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

class rsitmd_retrieval_eval_text(Dataset):
    def __init__(self, transform, img_dir, json_dir, split, max_words=30):
            '''
            img_dir (string): Root directory of images
            json_dir (string): Path to the JSON file
            split (string): Dataset split (e.g. test)
            max_words (int): Maximum number of words in a caption
            '''
            self.transform = transform
            self.img_dir = img_dir
            self.json_dir = json_dir
            self.split = split
            self.max_words = max_words

            self.image = []
            self.text = []
            self.txt2img = {}
            self.img2txt = {}
            self.max_words=max_words
            self._load_annotations()

    def _load_annotations(self):
        datasets = json.load(open(self.json_dir, "r"))
        txt_id = 0
        img_id_counter = 0
        for image in datasets['images']:
            if img_id_counter >= 450:
                break
            if image['split'] == self.split:
                self.image.append(image['filename'])
                self.img2txt[img_id_counter] = []
                for text in image['sentences']:
                    # caption = self._pre_caption(text['raw'])
                    caption = pre_caption(text['raw'],self.max_words)
                    self.text.append(caption)
                    self.img2txt[img_id_counter].append(txt_id)
                    # self.txt2img[txt_id] = img_id_counter
                    self.txt2img[txt_id] = []
                    self.txt2img[txt_id].append(img_id_counter)
                    txt_id += 1
                img_id_counter += 1

    def _pre_caption(self, caption):
        words = caption.split()
        if len(words) > self.max_words:
            words = words[:self.max_words]
        return ' '.join(words).capitalize()

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        if index not in self.txt2img:
            raise KeyError(f"Index {index} not found in txt2img")

        caption = self.text[index]
        return caption, index