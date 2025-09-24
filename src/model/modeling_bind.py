import time
import torch
import torch.nn as nn
import lightning as L

from types import MethodType
from typing import List, Optional, Tuple, Union
from copy import deepcopy
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from segmentation_models_pytorch.losses import FocalLoss
from grapheme import graphemes
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from src.tokenizer.modeling_tokenizer import BINDTokenizer, SentenceTokenizer
from src.model.utils import apply_neftune
from src.metrics.ChfF import chrf_corpus


class BIND(nn.Module):
    def __init__(self, base_model_name='Qwen/Qwen3-0.6B', use_sliding_window=True, sliding_window=12, use_bntd=True, neftune_alpha=0, n_tokens_per_char=4, use_qlora=False, lora_r=16, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.base_model_config = AutoConfig.from_pretrained(base_model_name)
        self.base_model_config._attn_implementation = 'eager'
        self.base_model_config.use_cache = False
        if use_bntd:
            self.base_model_config.use_sliding_window = use_sliding_window
            self.base_model_config.sliding_window = sliding_window

        if use_qlora:
            print("use_qlora")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                quantization_config=quantization_config
            )
            base_model.train()
            if use_bntd:
                base_model.model.sliding_window = sliding_window
                if 'qwen3' in base_model_name.lower():
                    base_model.model.forward = MethodType(qwen3_forward, base_model.model)
                    print('use full attn qwen3')
                elif 'gemma-3' in base_model_name.lower():
                    base_model.model.forward = MethodType(gemma3_forward, base_model.model)
                    print('use full attn gemma3')
                else:
                    raise ValueError('full attn model not found.')
            if neftune_alpha:
                base_model = apply_neftune(base_model, neftune_alpha)   
                print('neftune applied')  
            base_model = prepare_model_for_kbit_training(base_model)
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM"
            )
            base_model = get_peft_model(base_model, lora_config)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, config=self.base_model_config)
            base_model.train()
            if use_bntd:
                base_model.model.sliding_window = sliding_window
                if 'qwen3' in base_model_name.lower():
                    base_model.model.forward = MethodType(qwen3_forward, base_model.model)
                    print('use full attn qwen3')
                elif 'gemma-3' in base_model_name.lower():
                    base_model.model.forward = MethodType(gemma3_forward, base_model.model)
                    print('use full attn gemma3')
                else:
                    raise ValueError('full attn model not found.')
            if neftune_alpha:
                base_model = apply_neftune(base_model, neftune_alpha)   
                print('neftune applied') 

        self.model = base_model

        # ---- Detect Head 추가 ----
        hidden_size = self.model.config.hidden_size
        self.detect_window = nn.Conv1d(hidden_size, hidden_size, kernel_size=n_tokens_per_char, stride=n_tokens_per_char)
        self.detect_head = nn.Linear(hidden_size, 2)  # binary detection
        

    def set_tokenizer(self, bind_tokenizer: BINDTokenizer):
        self.tokenizer = bind_tokenizer


    # def forward(self, sentence_noisy, sentence=None, pred=False):
    #     output_ids = None
    #     input_ids, attention_mask, token_type_ids = self.tokenizer.batch_encode_char(sentence_noisy, self.tokenizer.input_chars_dict)

    #     if sentence is not None:
    #         output_ids, output_attention_mask, output_token_type_ids = self.tokenizer.batch_encode_char(sentence, self.tokenizer.target_chars_dict)
            
    #         correct_ids = output_ids.clone().to('cuda')
    #         correct_ids[output_token_type_ids == 0] = -100

    #         detect_labels = self.tokenizer.get_batch_detect_label(sentence, sentence_noisy).type_as(output_ids).to('cuda')

    #     input_ids = input_ids.to('cuda')
    #     attention_mask = attention_mask.to('cuda')

    #     outputs = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         output_hidden_states=True,
    #     )

    #     logits = outputs.logits
    #     hidden_states = outputs.hidden_states[-1]  # 마지막 layer hidden state [B, L, H]
    #     detect_logits = self.detect_window(hidden_states[:,:-2,:].transpose(1,2)).transpose(1,2)
    #     detect_logits = self.detect_head(detect_logits)
    #     detect_pred = detect_logits.argmax(-1).detach().cpu()

    #     loss = None
    #     if sentence is not None:
    #         correct_loss = nn.CrossEntropyLoss(reduction='mean')(
    #             logits[:, :-1, :].reshape(-1, self.model.config.vocab_size),
    #             correct_ids[:, 1:].reshape(-1),
    #         )

            
    #         detect_loss = FocalLoss('multiclass', ignore_index=-100)(
    #             detect_logits.reshape(-1, 2),
    #             detect_labels.reshape(-1)
    #         )
        
    #         loss = correct_loss + detect_loss

    #     # -------- Prediction --------
    #     pred_ids, sentence_denoised = [], []
    #     if pred:
    #         for idx in range(input_ids.shape[0]):
    #             input_ids_row = input_ids[idx].detach().cpu().tolist()[1:-1]
    #             pred_ids_row = logits[idx][:-2].argmax(-1).detach().cpu().tolist()
    #             token_type_ids_row = token_type_ids[idx].detach().cpu().tolist()[1:-1]

    #             pred_ids.append(pred_ids_row)
    #             sentence_denoised_row = self.tokenizer.decode_char(pred_ids_row, token_type_ids_row, input_ids_row, False)
    #             sentence_denoised_row_filtered = []
    #             for char_noisy, char_pred, detected_yn in zip(graphemes(sentence_noisy[idx]), graphemes(sentence_denoised_row), detect_pred[idx]):
    #                 if detected_yn==0:
    #                     sentence_denoised_row_filtered.append(char_noisy)
    #                 elif detected_yn==1:
    #                     sentence_denoised_row_filtered.append(char_pred)
    #             sentence_denoised_row_filtered = ''.join(sentence_denoised_row_filtered)

    #             sentence_denoised.append(sentence_denoised_row_filtered)

    #     return loss, logits, pred_ids, sentence_denoised

    def forward(self, sentence_noisy, sentence=None, pred=False):
        output_ids = None
        input_ids, attention_mask, token_type_ids = self.tokenizer.batch_encode_char(sentence_noisy, self.tokenizer.input_chars_dict)

        if sentence is not None:
            output_ids, output_attention_mask, output_token_type_ids = self.tokenizer.batch_encode_char(sentence, self.tokenizer.target_chars_dict)
            
            correct_ids = output_ids.clone().to('cuda')
            detect_labels = correct_ids.clone()
            correct_ids[output_token_type_ids == 0] = -100

            # detect_labels = self.tokenizer.get_batch_detect_label(sentence, sentence_noisy).type_as(output_ids).to('cuda')
            detect_labels[input_ids==output_ids] = 0
            detect_labels[input_ids!=output_ids] = 1

        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1][:,:-1,:]
        detect_logits = self.detect_head(hidden_states)

        loss = None
        if sentence is not None:
            correct_loss = nn.CrossEntropyLoss(reduction='mean')(
                logits[:, :-1, :].reshape(-1, self.model.config.vocab_size),
                correct_ids[:, 1:].reshape(-1),
            )

            detect_loss = FocalLoss('multiclass', ignore_index=-100)(
                detect_logits.reshape(-1, 2),
                detect_labels[:, 1:].reshape(-1)
            )
        
            loss = correct_loss + detect_loss

        # -------- Prediction --------
        pred_ids, sentence_denoised = [], []
        if pred:
            for idx in range(input_ids.shape[0]):
                input_ids_row = input_ids[idx].detach().cpu().tolist()[1:-1]
                pred_ids_row = logits[idx][:-2].argmax(-1).detach().cpu().tolist()
                token_type_ids_row = token_type_ids[idx].detach().cpu().tolist()[1:-1]

                pred_ids.append(pred_ids_row)
                sentence_denoised_row = self.tokenizer.decode_char(pred_ids_row, token_type_ids_row, input_ids_row, False)
                sentence_denoised.append(sentence_denoised_row)

        return loss, logits, pred_ids, sentence_denoised

def gemma3_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    attention_mask = _prepare_4d_attention_mask(attention_mask, self.dtype)
    
    # Create the masks
    attention_mask_mapping = {
        "full_attention": attention_mask,
    }
    sliding_attention_mask = torch.tril(torch.triu(attention_mask, diagonal=-self.sliding_window), diagonal=self.sliding_window)
    attention_mask_mapping["sliding_attention"] = sliding_attention_mask

    # embed positions
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
    position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            attention_mask=attention_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **flash_attn_kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

    
def qwen3_forward(
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
        hidden_states=[hidden_states]
    )


class LitBIND(L.LightningModule):
    def __init__(
        self,
        base_model_name='Qwen/Qwen3-0.6B',
        use_sliding_window=True,
        sliding_window=12,
        lr=5e-5,
        epochs=10,
        use_bntd=True,
        use_qlora=False,
        lora_r=16,
        lora_alpha=32,
        inference_sentence_min_length=32,
        inference_sentence_max_length=64,
        inference_sentence_n_overlap=3,
        n_tokens_per_char=4,
        input_chars='',
        target_chars='',
        neftune_alpha=0
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.lr = lr
        self.epochs = epochs
        self.use_bntd = use_bntd
        self.inference_sentence_min_length = inference_sentence_min_length
        self.inference_sentence_max_length = inference_sentence_max_length
        self.inference_sentence_n_overlap = inference_sentence_n_overlap

        self.bind = BIND(
            base_model_name=base_model_name,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            use_bntd=use_bntd,
            use_qlora=use_qlora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            neftune_alpha=neftune_alpha,
            n_tokens_per_char=n_tokens_per_char
        )
        bind_tokenizer = BINDTokenizer(base_tokenizer_name=base_model_name, n_tokens_per_char=n_tokens_per_char, input_chars=input_chars, target_chars=target_chars)
        self.bind.set_tokenizer(bind_tokenizer)
        self.sentence_tokenizer = SentenceTokenizer(
            min_length=inference_sentence_min_length,
            max_length=inference_sentence_max_length,
            n_overlap=inference_sentence_n_overlap,
            roll=False
        )

    def forward(self, batch, pred):
        loss, logits, pred_ids, sentence_denoised = self.bind.forward(
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
        sentences_denoised, times = self.predict_step(batch, batch_idx)
        score = chrf_corpus(sentences_denoised, batch['sentence'])['f1']
        self.log('valid_score', score, batch_size=len(batch['sentence_noisy']))
        return score
    
    def predict_step(self, batch, batch_idx):
        if self.inference_sentence_n_overlap > 1:
            sentences_noisy = batch['sentence_noisy']
            sentences_denoised = []
            times = []
            for sentence_noisy in sentences_noisy:
                start = time.time()
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
                end = time.time()
                times.append(end-start)
        else:
            sentences_noisy = batch['sentence_noisy']
            sentences_denoised = []
            for sentence_noisy in sentences_noisy:
                start = time.time()
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
                end = time.time()
                times.append(end-start)
        return sentences_denoised, times
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,           # 또는 2e-4
        )
        # return optimizer

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