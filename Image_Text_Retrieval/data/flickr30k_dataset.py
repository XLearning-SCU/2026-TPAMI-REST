import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from PIL import Image

from data.utils import pre_caption
    
class flickr30k_retrieval_eval_image(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
        #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        # download_url(urls[split],ann_root)
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                # self.txt2img[txt_id] = img_id
                self.txt2img[txt_id] = []
                self.txt2img[txt_id].append(img_id)
                txt_id += 1

        # print(f"txt2img has {len(self.txt2img)} entries")
        # print(f"img2txt has {len(self.img2txt)} entries")
        # # 打印字典的部分内容示例
        # if len(self.txt2img) > 0:
        #     print(f"txt2img sample: {list(self.txt2img.items())[:5]}")
        # if len(self.img2txt) > 0:
        #     print(f"img2txt sample: {list(self.img2txt.items())[:5]}")
        # exit(-1)
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index    
    
class flickr30k_retrieval_eval_text(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                # self.txt2img[txt_id] = img_id
                self.txt2img[txt_id] = []
                self.txt2img[txt_id].append(img_id)
                txt_id += 1

     
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):    
        caption = self.text[index]
        
        return caption, index    