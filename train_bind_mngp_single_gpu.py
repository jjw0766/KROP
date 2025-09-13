import os
import yaml
import argparse
from dotenv import load_dotenv

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from src.model.modeling_bind import LitBIND
from src.data.dataset import get_mngp_train_dataloader, get_mngp_dev_dataloader


def main(args):
    """
    Main training function that uses parsed arguments.
    """
    load_dotenv()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    L.seed_everything(args.seed)

    train_dl = get_mngp_train_dataloader(
        args.dataset_name,
        batch_size=args.mini_batch_size,
        max_length=args.train_max_length
    )
    dev_dl = get_mngp_dev_dataloader(
        args.dataset_name,
        batch_size=args.mini_batch_size,
        max_length=args.valid_max_length
    )


    lit_bind = LitBIND(
        base_model_name=args.base_model_name,
        lr=args.learning_rate,
        epochs=args.epochs,
        use_bntd=args.use_bntd,
        inference_sentence_max_length=args.inference_sentence_max_length,
        inference_sentence_min_length=args.inference_sentence_min_length,
        inference_sentence_n_overlap=args.inference_sentence_n_overlap,
        n_tokens_per_char=args.n_tokens_per_char,
        sliding_window=args.sliding_window,
        input_chars=args.input_chars,
        target_chars=args.target_chars,
        neftune_alpha=args.neftune_alpha
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/bind',
        filename=f"{args.dataset_name.split('/')[-1]}-mngp-{args.base_model_name.split('/')[-1]}-{args.prefix}" + "-{epoch:02d}-{valid_loss:.4f}",
        monitor='valid_loss',
        mode='min',
        save_weights_only=True,
        save_top_k=1,
    )

    trainer = L.Trainer(
        callbacks=[checkpoint_callback],
        precision='bf16',
        max_epochs=args.epochs,
        accumulate_grad_batches=args.n_batch
    )

    trainer.fit(lit_bind, train_dl, dev_dl)


def setup_parser():
    """
    Sets up the argument parser.
    """
    parser = argparse.ArgumentParser(description="Train a LitBIND model.")

    # General and hardware arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--cuda_visible_devices', type=str, default="1", help='CUDA device(s) to make visible.')

    # Data and model arguments
    parser.add_argument('--dataset_name', type=str, default='jwengr/C-LLM', help='Hugging Face dataset name.')
    parser.add_argument('--base_model_name', type=str, default='Qwen/Qwen3-0.6B-Base', help='Hugging Face base model name.')
    parser.add_argument('--n_tokens_per_char', type=int, default=4, help='n_tokens_per_char')
    parser.add_argument('--input_chars', type=str, default='', help='Target characters for the model.')
    parser.add_argument('--target_chars', type=str, default='', help='Target characters for the model.')

    # Training hyperparameters
    parser.add_argument('--mini_batch_size', type=int, default=16, help='Mini-batch size for training.')
    parser.add_argument('--n_batch', type=int, default=2, help='Number of gradient accumulation batches.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--use_bntd', type=bool, default=True, help='Whether to use BNTD in the model.')
    parser.add_argument('--sliding_window', type=int, default=12, help='sliding window size')
    parser.add_argument('--neftune_alpha', type=float, default=0, help='Whether to use neftune in the model.')

    # Text processing arguments
    parser.add_argument('--train_max_length', type=int, default=128, help='Max sequence length for training.')
    parser.add_argument('--valid_max_length', type=int, default=128, help='Max sequence length for validation.')
    parser.add_argument('--inference_sentence_max_length', type=int, default=64, help='Max sentence length for inference.')
    parser.add_argument('--inference_sentence_min_length', type=int, default=32, help='Min sentence length for inference.')
    parser.add_argument('--inference_sentence_n_overlap', type=int, default=3, help='Number of sentences to overlap during inference.')

    parser.add_argument('--prefix', type=str, default='')

    return parser


def load_yaml_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

if __name__ == "__main__":
    parser = setup_parser()
    # yaml config 불러오기
    import sys
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        cfg = load_yaml_config(sys.argv[1])
        args = parser.parse_args([])
        for k, v in cfg.items():
            setattr(args, k, v)
    else:
        args = parser.parse_args()
    main(args)