import os
import argparse
import yaml
from dotenv import load_dotenv

import lightning as L
import pandas as pd
from tqdm.auto import tqdm

from src.model.modeling_char_encoder import LitCharEncoder
from src.data.dataset import get_test_dataloader
from src.metrics.metric import calculate_metric


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
    lit_char_encoder = LitCharEncoder.load_from_checkpoint(
        args.checkpoint,
        base_model_name=config["base_model_name"],
        space_token=config["space_token"],
        unk_token=config["unk_token"],
        pad_token=config["pad_token"],
        lr=float(config["learning_rate"]),
        epochs=config["epochs"],
        inference_sentence_max_length=config["inference_sentence_max_length"],
        inference_sentence_min_length=config["inference_sentence_min_length"],
        inference_sentence_n_overlap=config["inference_sentence_n_overlap"],
        input_chars=config.get("input_chars", ""),
        target_chars=config.get("target_chars", "")
    )

    # Trainer for inference
    trainer = L.Trainer()
    preds = trainer.predict(lit_char_encoder, test_dl)

    # Collect predictions
    prediction = []
    for pred in tqdm(preds):
        prediction.extend(pred)

    # Build dataframe
    categories, inputs, true = [], [], []
    for batch in test_dl:
        true.extend(batch["sentence"])
        inputs.extend(batch["sentence_noisy"])
        if batch.get("category") is None:
            category = "none"
        else:
            category = batch["category"]
        categories.extend(category)

    result_df = pd.DataFrame({
        "input": inputs,
        "pred": prediction,
        "true": true,
        "category": categories,
    })

    # Calculate metrics
    for cat in set(result_df["category"]):
        cat_df = result_df[result_df["category"] == cat].copy()
        result, result_list = calculate_metric(
            cat_df["input"].tolist(),
            cat_df["true"].tolist(),
            cat_df["pred"].tolist()
        )
        print(cat, result)

    # Save results
    if args.output is not None:
        result_df.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")


def setup_parser():
    parser = argparse.ArgumentParser(description="Inference for LitBIND model")
    parser.add_argument("--config", type=str, default="train_config.yaml",
                        help="Path to YAML config file (same as training)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--cuda_visible_devices", type=str, default="0",
                        help="CUDA device(s) to make visible")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Path to save prediction results")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
