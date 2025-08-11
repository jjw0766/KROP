import os
import random
import torch
import datasets

from glob import glob
from torch.utils.data import Dataset, DataLoader
from konoise import NoiseGenerator

from src.tokenizer.modeling_tokenizer import SentenceTokenizer

class MABSATrainDataset(Dataset):
    def __init__(self, filename):
        self.genertor = NoiseGenerator()
        self.noise_methods = ['change-vowels', 'palatalization', 'linking', 'liquidization', 'nasalization', 'assimilation']
        self.sentences = []
        self.sentences_noisy = []

        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for sentence in lines:
                sentence = sentence.strip()
                sentence = sentence.split('####')[0]
                sentence_noisy_all = sentence
                for noise_method in self.noise_methods:
                    sentence_noisy_all = self.genertor.generate(sentence_noisy_all, noise_method, 1)[0][0]
                if sentence == sentence_noisy_all:
                    continue
                self.sentences.append(sentence)   
                self.sentences_noisy.append(sentence_noisy_all)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'sentence_noisy': self.sentences_noisy[idx]
        }
    
class TrainValCollateFn:
    def __init__(self, max_length, mode='train'):
        self.max_length=max_length
        self.mode=mode

    def sample(self, sentence_noisy, sentence, mode):
        if mode=='train':
            if self.max_length and (len(sentence_noisy) > self.max_length):
                start_indices = [i for i, char in enumerate(sentence_noisy) if char.isspace() and (i <= len(sentence_noisy) - self.max_length)]            
                if start_indices:
                    start_index = random.choice(start_indices)
                else:
                    start_index = random.randint(0, len(sentence_noisy) - self.max_length)
                sentence_noisy = sentence_noisy[start_index:start_index + self.max_length]
                sentence = sentence[start_index:start_index + self.max_length]
        elif mode=='valid':
            sentence_noisy = sentence_noisy[:self.max_length]
            sentence = sentence[:self.max_length]
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


def get_m_absa_train_dataloader(filename, batch_size, max_length):
    ds = MABSATrainDataset(filename)
    return DataLoader(ds, batch_size=batch_size, collate_fn=TrainValCollateFn(max_length=max_length))

def get_m_absa_dev_test_dataloader(dataset_name, **kwargs):
    ds = datasets.load_dataset(dataset_name)
    dev_ds, test_ds = ds['dev'], ds['test']
    return DataLoader(dev_ds, **kwargs), DataLoader(test_ds, **kwargs)

def get_train_dataloader(dataset_name, batch_size, max_length):
    ds = datasets.load_dataset(dataset_name)
    train_ds = ds['train']
    return DataLoader(train_ds, batch_size=batch_size, collate_fn=TrainValCollateFn(max_length=max_length, mode='train'))

def get_dev_dataloader(dataset_name, batch_size, max_length):
    ds = datasets.load_dataset(dataset_name)
    dev_ds = ds['dev']
    return DataLoader(dev_ds, batch_size=batch_size, collate_fn=TrainValCollateFn(max_length=max_length, mode='valid'))

def get_test_dataloader(dataset_name, batch_size):
    ds = datasets.load_dataset(dataset_name)
    dev_ds = ds['test']
    return DataLoader(dev_ds, batch_size=batch_size)
