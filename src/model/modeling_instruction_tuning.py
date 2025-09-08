import lightning as L
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

from src.model.utils import apply_neftune

class LitInstructionModel(L.LightningModule):
    def __init__(
        self,
        base_model_name='Qwen/Qwen3-0.6B',
        use_qlora=True,
        lr=5e-5,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        epochs=10,
        neftune_alpha=0
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.use_qlora = use_qlora
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
    
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
                quantization_config=quantization_config
            )
            model.train()
            if neftune_alpha:
                model = apply_neftune(model, neftune_alpha)   
                print('neftune applied') 
            model = prepare_model_for_kbit_training(model)
    
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name, 
            )
            model.train()
            if neftune_alpha:
                model = apply_neftune(model, neftune_alpha)   
                print('neftune applied') 
        self.model = model
        

    def get_prompt(self, sentence_noisy, sentence, mode='train'):
        if mode=='train':
            messages = [
                {"role": "user", "content": f"Please decode the obfuscated text. Text: {sentence_noisy}"},
                {"role": "assistant", "content": sentence}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
            )
        elif mode=='inference':
            messages = [
                {"role": "user", "content": f"Please decode the obfuscated text. Text: {sentence_noisy}"},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
            )
        return prompt
    
    def batch_tokenize(self, batch, mode):
        prompts = []
        for sentence_noisy, sentence in zip(batch['sentence_noisy'], batch['sentence']):
            prompt = self.get_prompt(sentence_noisy, sentence, mode)
            prompts.append(prompt)
        inputs = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding=True, padding_side='left')
        return inputs

    def forward(self, batch):
        inputs = self.batch_tokenize(batch, mode='train')
        outputs = self.model(
            input_ids=inputs['input_ids'].to('cuda'),
            attention_mask=inputs['attention_mask'].to('cuda'),
            labels=inputs['input_ids'].to('cuda')
        )
        return outputs.loss, outputs.logits

    
    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.log('train_loss', loss, batch_size=len(batch['sentence_noisy']), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits = self(batch)
        self.log('valid_loss', loss, batch_size=len(batch['sentence_noisy']))
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs = self.batch_tokenize(batch, mode='inference')
        outputs = self.model.generate(
            input_ids = inputs['input_ids'].to('cuda'),
            attention_mask = inputs['attention_mask'].to('cuda')
        )
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded
        
    
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