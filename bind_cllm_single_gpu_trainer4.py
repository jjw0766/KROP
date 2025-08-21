import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="4"


from dotenv import load_dotenv
load_dotenv()

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint

from src.model.modeling_bind import LitBIND
from src.data.dataset import get_train_dataloader, get_dev_dataloader, get_test_dataloader

SEED=42
DATASET_NAME = 'jwengr/C-LLM'
MINI_BATCH_SIZE=2
N_BATCH = 16
BASE_MODEL_NAME='Qwen/Qwen3-4B-Base'
EPOCHS=10
LEARNING_RATE = 1e-4
USE_BNTD=True
TRAIN_MAX_LENGTH=128
VALID_MAX_LENGTH=128
INFERENCE_SENTENCE_MAX_LENGTH=64
INFERENCE_SENTENCE_MIN_LENGTH=32
INFERENCE_SENTENCE_N_OVERLAP=3

L.seed_everything(SEED)

train_dl = get_train_dataloader(DATASET_NAME, batch_size=MINI_BATCH_SIZE, max_length=TRAIN_MAX_LENGTH)
dev_dl = get_dev_dataloader(DATASET_NAME, batch_size=MINI_BATCH_SIZE, max_length=VALID_MAX_LENGTH)
test_dl = get_test_dataloader(DATASET_NAME, batch_size=MINI_BATCH_SIZE)

lit_bind = LitBIND(
    base_model_name=BASE_MODEL_NAME,
    lr=LEARNING_RATE,
    epochs=EPOCHS,
    use_bntd=USE_BNTD,
    inference_sentence_max_length=INFERENCE_SENTENCE_MAX_LENGTH,
    inference_sentence_min_length=INFERENCE_SENTENCE_MIN_LENGTH,
    inference_sentence_n_overlap=INFERENCE_SENTENCE_N_OVERLAP
)

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/bind',
    filename=f"{DATASET_NAME.split('/')[1]}-{BASE_MODEL_NAME.split('/')[1]}"+"-{epoch:02d}-{valid_loss:.4f}",
    monitor='valid_loss',
    mode='min',
    save_weights_only=True,
    save_top_k=3,
)

trainer = L.Trainer(
    callbacks=[checkpoint_callback],
    precision='bf16',
    max_epochs=EPOCHS,
    accumulate_grad_batches=N_BATCH
)

trainer.fit(lit_bind, train_dl, dev_dl)