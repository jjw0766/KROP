import time
import lightning as L
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import T5ForConditionalGeneration, BitsAndBytesConfig, AutoTokenizer

from src.model.utils import apply_neftune
from src.metrics.ChfF import chrf_corpus

class LitT5(L.LightningModule):
    def __init__(
        self,
        base_model_name='google/byt5-small',
        use_qlora=True,
        epochs=10,
        lr=5e-5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.lr = lr
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if use_qlora:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
    
            model = T5ForConditionalGeneration.from_pretrained(
                base_model_name, 
                quantization_config=quantization_config
            )
            model = prepare_model_for_kbit_training(model)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            model = get_peft_model(model, lora_config)
        else:
            model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.model = model

    def forward(self, batch):
        inputs = self.tokenizer(batch['sentence_noisy'], padding="longest", return_tensors="pt").to('cuda')
        labels = self.tokenizer(batch['sentence'], padding="longest", return_tensors="pt").input_ids.to('cuda')
        loss = self.model(**inputs, labels=labels).loss
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, batch_size=len(batch['sentence_noisy']), on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        sentences_denoised, times = self.predict_step(batch, batch_idx)
        score = chrf_corpus(sentences_denoised, batch['sentence'])['f1']
        self.log('valid_score', score, batch_size=len(batch['sentence_noisy']))
        return score
    
    def predict_step(self, batch, batch_idx):
        times = []
        start = time.time()
        inputs = self.tokenizer(batch['sentence_noisy'], padding="longest", return_tensors="pt").to('cuda')
        outputs = self.model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_new_tokens = inputs['input_ids'].shape[1]*2,
            do_sample=False
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        end = time.time()
        times.append(end-start)
        return decoded, times

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