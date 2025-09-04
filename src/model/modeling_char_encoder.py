import torch
import torch.nn as nn
import lightning as L

from types import MethodType
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoModelWithLMHead
from segmentation_models_pytorch.losses import FocalLoss

from src.tokenizer.modeling_tokenizer import CharEncoderTokenizer, SentenceTokenizer


class CharEncoder(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.model = AutoModelWithLMHead.from_pretrained(base_model_name)
        hidden_size = self.model.config.hidden_size
        self.detect_head = nn.Linear(hidden_size, 2)

    def set_tokenizer(self, tokenizer: CharEncoderTokenizer):
        self.tokenizer = tokenizer

    def forward(self, sentence_noisy, sentence=None, pred=False):
        output_ids = None
        input_ids, attention_mask, token_type_ids = self.tokenizer.batch_encode_char(sentence_noisy)
        if sentence is not None:
            output_ids, *_ = self.tokenizer.batch_encode_char(sentence)

            correct_ids = output_ids.clone().to('cuda')
            correct_ids[token_type_ids == 0] = -100

            detect_ids = ((input_ids != output_ids) * 1).type_as(output_ids).to('cuda')
            detect_ids[token_type_ids == 0] = -100  # loss ignore index를 위해 masking

        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        loss = None
        correct_loss = None
        detect_loss = None
        if sentence is not None:
            correct_loss = nn.CrossEntropyLoss(reduction='mean')(
                logits.reshape(-1, self.model.config.vocab_size),
                correct_ids.reshape(-1),
            )

            detect_logits = self.detect_head(hidden_states)  
            detect_loss = FocalLoss('multiclass', ignore_index=-100)(
                detect_logits.reshape(-1, 2),
                detect_ids.reshape(-1),
            )

            loss = correct_loss + detect_loss

        pred_ids = []
        sentence_denoised = []
        if pred:
            for idx in range(input_ids.shape[0]):
                pred_ids_row = logits[idx].argmax(-1).detach().cpu().tolist()
                token_type_ids_row = token_type_ids[idx].cpu().tolist()
                sentence_denoised.append(self.tokenizer.decode_char(pred_ids_row, token_type_ids_row, sentence_noisy[idx]))
                pred_ids.append(pred_ids_row)
        return loss, logits, pred_ids, sentence_denoised
    
    
class LitCharEncoder(L.LightningModule):
    def __init__(
        self,
        base_model_name,
        space_token='[SEP]',
        unk_token='[UNK]',
        pad_token='[PAD]',
        lr=5e-5,
        epochs=10,
        inference_sentence_min_length=32,
        inference_sentence_max_length=64,
        inference_sentence_n_overlap=3,
        target_chars=[]
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.lr = lr
        self.epochs = epochs
        self.inference_sentence_min_length = inference_sentence_min_length
        self.inference_sentence_max_length = inference_sentence_max_length
        self.inference_sentence_n_overlap = inference_sentence_n_overlap

        self.encoder = CharEncoder(base_model_name=base_model_name)
        encoder_tokenizer = CharEncoderTokenizer(base_tokenizer_name=base_model_name, space_token=space_token, unk_token=unk_token, pad_token=pad_token, target_chars=target_chars)
        self.encoder.set_tokenizer(encoder_tokenizer)
        self.sentence_tokenizer = SentenceTokenizer(
            min_length=inference_sentence_min_length,
            max_length=inference_sentence_max_length,
            n_overlap=inference_sentence_n_overlap,
            roll=False
        )

    def forward(self, batch, pred):
        loss, logits, pred_ids, sentence_denoised = self.encoder.forward(
            sentence_noisy=batch['sentence_noisy'],
            sentence=batch['sentence'],
            pred=pred
        )
        return loss, logits, pred_ids, sentence_denoised
    
    def training_step(self, batch, batch_idx):
        loss, logits, *_ = self(batch, pred=False)
        self.log('train_loss', loss, batch_size=len(batch['sentence_noisy']), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, *_ = self(batch, pred=False)
        self.log('valid_loss', loss, batch_size=len(batch['sentence_noisy']))
        return loss
    
    def predict_step(self, batch, batch_idx):
        if self.inference_sentence_n_overlap > 1:
            sentences_noisy = batch['sentence_noisy']
            sentences_denoised = []
            for sentence_noisy in sentences_noisy:
                sentence_denoised_chunks = []
                sentence_noisy_chunks = self.sentence_tokenizer.split_text(sentence_noisy)
                for sentence_noisy_chunk in sentence_noisy_chunks:
                    mini_batch = {
                        'sentence_noisy': [sentence_noisy_chunk],
                        'sentence': None
                    }
                    loss, logits, pred_ids, sentence_denoised_chunk = self(mini_batch, pred=True)
                    sentence_denoised_chunks.append(sentence_denoised_chunk[0])
                sentence_denoised = ''.join(sentence_denoised_chunks)
                sentences_denoised.append(sentence_denoised)
        else:
            sentences_noisy = batch['sentence_noisy']
            sentences_denoised = []
            for sentence_noisy in sentences_noisy:
                sentence_denoised_chunks_overlapped = []
                sentence_noisy_chunks = self.sentence_tokenizer.split_text(sentence_noisy)
                sentence_noisy_chunks_overlapped = self.sentence_tokenizer.overlap(sentence_noisy_chunks)
                for start_idx, end_idx, sentence_noisy_chunk in sentence_noisy_chunks_overlapped:
                    mini_batch = {
                        'sentence_noisy': [sentence_noisy_chunk],
                        'sentence': None
                    }
                    loss, logits, pred_ids, sentence_denoised_chunk = self(mini_batch, pred=True)
                    sentence_denoised_chunks_overlapped.append((start_idx, end_idx, sentence_denoised_chunk[0]))
                sentence_denoised = self.sentence_tokenizer.decode_overlap(sentence_denoised_chunks_overlapped)
                sentences_denoised.append(sentence_denoised)
        return sentences_denoised
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,           # 또는 2e-4
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,          # 총 epoch 수
            eta_min=1e-6        # 최소 learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 매 epoch마다 갱신
                "frequency": 1,
            }
        }