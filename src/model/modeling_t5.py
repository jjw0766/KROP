import lightning as L
import torch

from transformers import T5ForConditionalGeneration, AutoTokenizer

from src.model.utils import apply_neftune

class LitT5(L.LightningModule):
    def __init__(
        self,
        base_model_name='google/byt5-small',
        epochs=10,
        lr=5e-5
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.lr = lr
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)

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
        loss = self(batch)
        self.log('valid_loss', loss, batch_size=len(batch['sentence_noisy']))
        return loss
    
    def predict_step(self, batch, batch_idx):
        inputs = self.tokenizer(batch['sentence_noisy'], padding="longest", return_tensors="pt").to('cuda')
        outputs = self.model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_new_tokens = inputs['input_ids'].shape[1]*2,
            do_sample=False
        )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
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