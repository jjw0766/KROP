import os
import math
import random
import torch
import datasets

from glob import glob
from grapheme import graphemes, slice
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.modeling_tokenizer import SentenceTokenizer
    
class TrainValCollateFn:
    def __init__(self, max_length, mode='train'):
        self.max_length=max_length
        self.mode=mode

    def sample(self, sentence_noisy, sentence, mode):
        if mode=='train':
            if self.max_length and (len(list(graphemes(sentence_noisy))) > self.max_length):
                start_indices = [i for i, char in enumerate(graphemes(sentence_noisy)) if char.isspace() and (i <= len(list(graphemes(sentence_noisy))) - self.max_length)]            
                if start_indices:
                    start_index = random.choice(start_indices)
                else:
                    start_index = random.randint(0, len(list(graphemes(sentence_noisy))) - self.max_length)
                # sentence_noisy = sentence_noisy[start_index:start_index + self.max_length]
                # sentence = sentence[start_index:start_index + self.max_length]
                sentence_noisy = slice(sentence_noisy, start_index, start_index + self.max_length)
                sentence = slice(sentence, start_index, start_index + self.max_length)
                
        elif mode=='valid':
            # sentence_noisy = sentence_noisy[:self.max_length]
            # sentence = sentence[:self.max_length]
            sentence_noisy = slice(sentence_noisy, 0, self.max_length)
            sentence = slice(sentence, 0, self.max_length)
        return sentence_noisy, sentence

    def __call__(self, examples):
        sentences_noisy = []
        sentences = []
        for example in examples:
            sentence_noisy, sentence = self.sample(example['sentence_noisy'], example['sentence'], mode=self.mode)
            sentences_noisy.append(sentence_noisy)
            sentences.append(sentence)
        return {
            'sentence_noisy': sentences_noisy,
            'sentence': sentences
        }
    
class MaskedNextGraphmePredictionTrainValCollateFn:
    def __init__(self, max_length, mode='train', sample_ratio=0.15, mask_token='▁'): #"▁"(U+2581)
        self.max_length=max_length
        self.mode=mode
        self.sample_ratio=sample_ratio
        self.mask_token=mask_token

    def sample(self, sentence, mode):
        if mode=='train':
            if self.max_length and (len(list(graphemes(sentence))) > self.max_length):
                start_indices = [i for i, char in enumerate(graphemes(sentence)) if char.isspace() and (i <= len(list(graphemes(sentence))) - self.max_length)]            
                if start_indices:
                    start_index = random.choice(start_indices)
                else:
                    start_index = random.randint(0, len(list(graphemes(sentence))) - self.max_length)
                sentence = slice(sentence, start_index, start_index + self.max_length)
            sentence_noisy = []
            for char in list(graphemes(sentence)):
                if random.random()<self.sample_ratio:
                    sentence_noisy.append(self.mask_token)
                else:
                    sentence_noisy.append(char)    
            sentence_noisy = ''.join(sentence_noisy)        
        elif mode=='valid':
            sentence = slice(sentence, 0, self.max_length)
            sentence_noisy = list(graphemes(sentence))
            interval = math.ceil((len(sentence_noisy)*self.sample_ratio))
            i = 0
            while i<len(list(graphemes(sentence))):
                sentence_noisy[i] = self.mask_token
                i += interval
            sentence_noisy = ''.join(sentence_noisy)

        return sentence_noisy, sentence

    def __call__(self, examples):
        sentences_noisy = []
        sentences = []
        for example in examples:
            sentence_noisy, sentence = self.sample(example['sentence'], mode=self.mode)
            sentences_noisy.append(sentence_noisy)
            sentences.append(sentence)
        return {
            'sentence_noisy': sentences_noisy,
            'sentence': sentences
        }

def get_train_dataloader(dataset_name, batch_size, max_length, category=None):
    ds = datasets.load_dataset(dataset_name)
    train_ds = ds['train']
    if category:
        train_ds = train_ds.filter(lambda example: example['category']==category)
    return DataLoader(train_ds, batch_size=batch_size, collate_fn=TrainValCollateFn(max_length=max_length, mode='train'), shuffle=True)

def get_dev_dataloader(dataset_name, batch_size, max_length, category=None):
    ds = datasets.load_dataset(dataset_name)
    dev_ds = ds['dev']
    if category:
        dev_ds = dev_ds.filter(lambda example: example['category']==category)
    return DataLoader(dev_ds, batch_size=batch_size)

def get_test_dataloader(dataset_name, batch_size, category=None):
    ds = datasets.load_dataset(dataset_name)
    test_ds = ds['test']
    if category:
        test_ds = test_ds.filter(lambda example: example['category']==category)
    return DataLoader(test_ds, batch_size=batch_size)

def get_mngp_train_dataloader(dataset_name, batch_size, max_length, category=None):
    ds = datasets.load_dataset(dataset_name)
    train_ds = ds['train']
    if category:
        train_ds = train_ds.filter(lambda example: example['category']==category)
    return DataLoader(train_ds, batch_size=batch_size, collate_fn=MaskedNextGraphmePredictionTrainValCollateFn(max_length=max_length, mode='train'), shuffle=True)

def get_mngp_dev_dataloader(dataset_name, batch_size, max_length):
    ds = datasets.load_dataset(dataset_name)
    dev_ds = ds['dev']
    return DataLoader(dev_ds, batch_size=batch_size, collate_fn=MaskedNextGraphmePredictionTrainValCollateFn(max_length=max_length, mode='valid'))