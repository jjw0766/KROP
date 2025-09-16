import os
import argparse
import yaml
from dotenv import load_dotenv

import lightning as L
import pandas as pd
from tqdm.auto import tqdm

from src.model.modeling_bind import LitBIND
from src.data.dataset import get_test_dataloader


def load_config(config_path: str):
    """Load YAML config file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    load_dotenv()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Load config
    config = load_config(args.config)

    # Test dataloader
    test_dl = get_test_dataloader(
        config["dataset_name"],
        batch_size=config["mini_batch_size"]
    )

    # Load checkpoint
    lit_bind = LitBIND.load_from_checkpoint(
        args.checkpoint,
        base_model_name=config["base_model_name"],
        lr=float(config["learning_rate"]),
        epochs=config["epochs"],
        use_bntd=config["use_bntd"],
        use_qlora=config['use_qlora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        sliding_window=config['sliding_window'],
        inference_sentence_max_length=config["inference_sentence_max_length"],
        inference_sentence_min_length=config["inference_sentence_min_length"],
        inference_sentence_n_overlap=config["inference_sentence_n_overlap"],
        n_tokens_per_char=config.get("n_tokens_per_char",4),
        input_chars=config.get("input_chars", ""),
        target_chars=config.get("target_chars", ""),
        strict=False
    )

    # Trainer for inference
    trainer = L.Trainer()
    preds = trainer.predict(lit_bind, test_dl)

    # Collect predictions
    prediction = []
    all_times = []
    for pred, times in tqdm(preds):
        prediction.extend(pred)
        all_times.extend(times)

    # Build dataframe
    categories, inputs, true = [], [], []
    for batch in test_dl:
        true.extend(batch["sentence"])
        inputs.extend(batch["sentence_noisy"])
        if batch.get("category") is None:
            category = ["none"]*len(batch["sentence"])
        else:
            category = batch["category"]
        categories.extend(category)

    result_df = pd.DataFrame({
        "input": inputs,
        "pred": prediction,
        "true": true,
        "category": categories,
        "times":all_times
    })


    # Save results
    base_name = os.path.basename(args.checkpoint)
    save_name = os.path.splitext(base_name)[0] + ".csv"
    os.makedirs('results', exist_ok=True)
    result_df.to_csv(f"results/bind_{save_name}", index=False)


def setup_parser():
    parser = argparse.ArgumentParser(description="Inference for LitBIND model")
    parser.add_argument("--config", type=str, default="train_config.yaml",
                        help="Path to YAML config file (same as training)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--cuda_visible_devices", type=str, default="0",
                        help="CUDA device(s) to make visible")

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
