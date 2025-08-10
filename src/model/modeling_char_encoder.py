import torch
import torch.nn as nn
import lightning as L

from types import MethodType
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoModelWithLMHead

from src.tokenizer.modeling_tokenizer import CharEncoderTokenizer


class CharEncoder(nn.Module):
    def __init__(self, base_model_name='klue/roberta-small'):
        super().__init__()
        self.model = AutoModelWithLMHead.from_pretrained(base_model_name)

    def set_tokenizer(self, tokenizer: CharEncoderTokenizer):
        self.tokenizer = tokenizer

    def forward(self, sentence_noisy, sentence=None, pred=False):
        output_ids = None
        input_ids, attention_mask, token_type_ids = self.tokenizer.batch_encode_char(sentence_noisy)
        if sentence is not None:
            output_ids, *_ = self.tokenizer.batch_encode_char(sentence)

        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        token_type_ids = token_type_ids.to('cuda')

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits

        loss = None
        if output_ids is not None:
            output_ids = output_ids.to('cuda')
            loss = nn.CrossEntropyLoss(reduction='mean')(
                logits.reshape(-1, self.model.config.vocab_size),
                output_ids.reshape(-1),
            )

        pred_ids = None
        sentence_denoised = []
        if pred:
            for idx in range(input_ids.shape[0]):
                pred_ids = self.pred_char_ids(logits[idx].detach(), input_ids[idx], token_type_ids[idx][1:-1]).detach().cpu().tolist()
                sentence_denoised.append(self.tokenizer.decode_char(pred_ids, token_type_ids[idx].tolist(), False))
        return loss, logits, pred_ids, sentence_denoised
    
class LitCharEncoder(L.LightningModule):
    def __init__(
        self,
        base_model_name='klue/roberta-small',
        space_token='[SEP]',
        target_language='kor',
        lr=5e-5,
        epochs=10,
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.lr = lr
        self.epochs = epochs

        self.encoder = CharEncoder(base_model_name=base_model_name)
        encoder_tokenizer = CharEncoderTokenizer(base_tokenizer_name=base_model_name, space_token=space_token, target_language=target_language)
        self.encoder.set_tokenizer(encoder_tokenizer)

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
        loss, logits, pred_ids, sentence_denoised = self(batch, pred=False)
        return sentence_denoised
    
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