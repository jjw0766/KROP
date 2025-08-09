import torch
import torch.nn as nn
import lightning as L

from types import MethodType
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast

from src.tokenizer.tokenizer import KropTokenizer


class Krop(nn.Module):
    def __init__(self, base_model_name='Qwen/Qwen3-0.6B', use_sliding_window=True, sliding_window=12):
        super().__init__()
        self.base_model_config = AutoConfig.from_pretrained(base_model_name)
        self.base_model_config._attn_implementation = 'eager'
        self.base_model_config.use_cache=False
        self.base_model_config.use_sliding_window = use_sliding_window
        self.base_model_config.sliding_window = sliding_window
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, config=self.base_model_config)
        base_model.model.forward = MethodType(model_forward, base_model.model)
        self.model = base_model

    def set_tokenizer(self, krop_tokenizer: KropTokenizer):
        self.tokenizer = krop_tokenizer
        self.cho_ids = nn.Parameter(torch.LongTensor(self.tokenizer.cho_ids), requires_grad=False)
        self.joong_ids = nn.Parameter(torch.LongTensor(self.tokenizer.joong_ids), requires_grad=False)
        self.jong_ids = nn.Parameter(torch.LongTensor(self.tokenizer.jong_ids), requires_grad=False)
        self.char_ids = nn.Parameter(torch.LongTensor(self.tokenizer.char_ids), requires_grad=False)
        
    def forward(self, sentence_noisy, sentence=None, task='jamo', pred=False):
        output_ids = None
        if task=='jamo':
            input_ids, attention_mask, token_type_ids = self.tokenizer.batch_encode_jamo(sentence)
            if sentence is not None:
                output_ids, *_ = self.tokenizer.batch_encode_jamo(sentence_noisy)
        elif task=='char':
            input_ids, attention_mask, token_type_ids = self.tokenizer.batch_encode_char(sentence)
            if sentence is not None:
                output_ids, *_ = self.tokenizer.batch_encode_char(sentence_noisy)

        input_ids = input_ids.to(self.cho_ids.device)
        attention_mask = attention_mask.to(self.cho_ids.device)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        loss = None
        if output_ids is not None:
            output_ids = output_ids.to(self.cho_ids.device)
            loss = nn.CrossEntropyLoss(reduction='mean')(
                logits[:, :-1, :].reshape(-1, self.model.config.vocab_size),
                output_ids[:,1:].reshape(-1),
            )

        pred_ids = None
        sentence_denoised = []
        if pred:
            if task=='jamo':
                pred_ids = self.pred_jamo_ids(logits.detach(), token_type_ids)
            elif task=='char':
                pred_ids = self.pred_char_ids(logits.detach(), token_type_ids)
            for idx in range(input_ids.shape[0]):
                if task=='jamo':
                    sentence_denoised.append(self.tokenizer.decode_jamo(pred_ids[idx].tolist(), token_type_ids[idx].tolist()))
                elif task=='char':
                    sentence_denoised.append(self.tokenizer.decode_char(pred_ids[idx].tolist(), token_type_ids[idx].tolist()))
        return loss, logits, pred_ids, sentence_denoised

    def pred_jamo_ids(
        self,
        logits,
        token_type_ids,
    ):
        pred_ids = token_type_ids.clone()
        logits_cho = logits[token_type_ids==1][:, self.cho_ids]
        logits_joong = logits[token_type_ids==2][:, self.joong_ids]
        logits_jong = logits[token_type_ids==3][:, self.jong_ids]
        pred_cho_ids = self.cho_ids[logits_cho.argmax(1)]
        pred_joong_ids = self.joong_ids[logits_joong.argmax(1)]
        pred_jong_ids = self.jong_ids[logits_jong.argmax(1)]
        pred_ids[token_type_ids==1] = pred_cho_ids
        pred_ids[token_type_ids==2] = pred_joong_ids
        pred_ids[token_type_ids==3] = pred_jong_ids
        return pred_ids

    def pred_char_ids(
        self,
        logits,
        token_type_ids
    ):
        pred_ids = token_type_ids.clone()
        logits_char = logits[token_type_ids==4]
        if not len(logits_char):
            return pred_ids
        logits_char_chunks = logits_char.split(3)
        pred_char_ids = []
        for logits_char_chunk in logits_char_chunks:
            logits_char_ids = torch.stack([
                logits_char_chunk[0][self.char_ids[:,0]], 
                logits_char_chunk[1][self.char_ids[:,1]], 
                logits_char_chunk[2][self.char_ids[:,2]]
            ], 1)
            pred_char_ids.extend( self.char_ids[ logits_char_ids.log_softmax(-1).sum(1).argmax() ] )
        pred_ids[token_type_ids==4] = torch.LongTensor(pred_char_ids).type_as(pred_ids)
        return pred_ids    
    
def model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values  = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs ,
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
    attention_mask = attention_mask == 0
    
    # Create the masks
    attention_mask_mapping = {
        "full_attention": attention_mask,
    }
    # The sliding window alternating layers are not always activated depending on the config
    if self.has_sliding_layers:
        sliding_attention_mask = torch.tril(torch.triu(attention_mask, diagonal=-self.sliding_window), diagonal=self.sliding_window)
        attention_mask_mapping["sliding_attention"] = sliding_attention_mask

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


class LitKrop(L.LightningModule):
    def __init__(
        self,
        base_model_name='Qwen/Qwen3-0.6B',
        use_sliding_window=True,
        sliding_window=12,
        lr=5e-5,
        task='jamo'
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.lr = lr
        self.task = task

        self.krop = Krop(
            base_model_name=base_model_name,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window
        )
        krop_tokenizer = KropTokenizer(base_tokenizer_name=base_model_name)
        self.krop.set_tokenizer(krop_tokenizer)

    def forward(self, batch, pred):
        if self.task=='jamo':
            loss, logits, pred_ids, sentence_denoised = self.krop.forward(
                sentence_noisy=batch['sentence_noisy'],
                sentence=batch['sentence'],
                task='jamo',
                pred=pred
            )
        elif self.task=='char':
            loss, logits, pred_ids, sentence_denoised = self.krop.forward(
                sentence_noisy=batch['sentence_noisy'],
                sentence=batch['sentence'],
                task='char',
                pred=pred
            )
        elif self.task=='char-jamo':
            loss_char, logits, pred_ids, sentence_denoised = self.krop.forward(
                sentence_noisy=batch['sentence_noisy'],
                sentence=batch['sentence'],
                task='char',
                pred=True
            )
            loss_jamo, logits, pred_ids, sentence_denoised = self.krop.forward(
                sentence_noisy=sentence_denoised,
                sentence=batch['sentence'],
                task='jamo',
                pred=pred
            )
            loss = loss_char+loss_jamo
        elif self.task=='jamo-char':
            loss_jamo, logits, pred_ids, sentence_denoised = self.krop.forward(
                sentence_noisy=batch['sentence_noisy'],
                sentence=batch['sentence'],
                task='jamo',
                pred=True
            )
            loss_char, logits, pred_ids, sentence_denoised = self.krop.forward(
                sentence_noisy=sentence_denoised,
                sentence=batch['sentence'],
                task='char',
                pred=pred
            )
            loss = loss_char+loss_jamo
        return loss, logits, pred_ids, sentence_denoised
    
    def training_step(self, batch, batch_idx):
        loss, logits, *_ = self(batch, pred=False)
        self.log('train_loss', loss, batch_size=len(batch['sentence_noisy']), on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        loss, logits, *_ = self(batch, pred=False)
        self.log('valid_loss', loss, batch_size=len(batch['sentence_noisy']))
    
    def predict_step(self, batch, batch_idx):
        loss, logits, pred_ids, sentence_denoised = self(batch, pred=False)
        return sentence_denoised
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr )