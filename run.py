import os
import os.path as osp
import glob
from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger
from datamodules import FLORESDataModule, BMLAMADataModule
from model import MultilingualModel
from utils import CustomCallback, CustomMetricTracker, CustomRichProgressBar

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(args):
    # Set seed
    L.seed_everything(args.seed, workers=True)

    # Set up wandb
    _i = args.output_dir.find("BS") # Split the output_dir to group and name based on batch size string
    wandb_logger = WandbLogger(
        project="multilingual-unlearning",
        group="/".join(args.output_dir.split("/")[1:_i]),
        name="/".join(args.output_dir.split("/")[_i:]),
        mode=args.wandb_mode,
    )

    # Load datamodule
    if args.task == "flores":
        dm = FLORESDataModule(args)
    elif args.task == "bmlama":
        dm = BMLAMADataModule(args)
    else:
        raise ValueError(f"Task {args.task} not supported.")

    # Load model
    if args.finetuned_model_path:
        print(f"Loading model from {args.finetuned_model_path}...")
        model = MultilingualModel.load_from_checkpoint(
            checkpoint_path=args.finetuned_model_path,
            hparams=args,
        )
    else:
        model = MultilingualModel(args)
    if args.torch_compile:
        model = torch.compile(model)

    # Callbacks
    callbacks = [
        CustomMetricTracker(args.output_dir),
        CustomRichProgressBar(),
    ]
    if not args.disable_checkpointing:
        cb = CustomCallback(args)
        callbacks.extend([
            cb.load_checkpoint_callback(),
            cb.load_early_stopping_callback(),
        ])

    trainer = L.Trainer(
        default_root_dir=args.output_dir,
        accelerator="gpu",
        devices="auto",
        strategy="auto",
        plugins=None,
        precision="16-mixed" if args.fp16 else "bf16-mixed" if args.bf16 else "32-true",
        max_epochs=args.epochs,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=args.logging_steps,
        val_check_interval=args.evaluation_steps,
        num_sanity_val_steps=0,
        deterministic=args.deterministic,
        logger=wandb_logger,
        reload_dataloaders_every_n_epochs=args.alternate_loader_every_n_epoch,
        enable_checkpointing=not args.disable_checkpointing,
        callbacks=callbacks,
    )

    if args.do_train:
        trainer.fit(model, dm)

    if args.do_eval or args.do_test:
        if args.ckpt_path:
            ckpt_path = osp.join(args.output_dir, args.ckpt_path)
        else:
            try:
                # Load the model with the smallest forget accuracy
                ckpt_path = sorted(glob.glob(osp.join(args.output_dir, "*.ckpt")))[0]
            except IndexError:
                ckpt_path = ""

        if osp.exists(ckpt_path):
            print(f"Loading the best model from {ckpt_path}...")
            model = MultilingualModel.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                hparams=args,
            )
            if args.torch_compile:
                model = torch.compile(model)
        else:
            print(f"Running evaluation without loading a checkpoint...")

        # Inference across multiple languages
        if args.do_eval:
            trainer.validate(model, dm)
        if args.do_test:
            trainer.test(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training")
    # Model arguments
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--method", type=str, default="original")
    parser.add_argument("--finetuned_model_path", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, default="")
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--task", type=str, default="flores")
    parser.add_argument("--forget_lang", type=str, nargs="+", default=["en"])
    parser.add_argument("--retain_lang", type=str, nargs="+", default=["en"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--forget_num", type=int, default=32)
    parser.add_argument("--forget_multiplier", type=int, default=1)
    parser.add_argument("--retain_multiplier", type=int, default=1)
    parser.add_argument("--alternate_loader_every_n_epoch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mini_bmlama", action="store_true")
    # Training arguments
    parser.add_argument("--output_dir", type=str, default=".checkpoints/")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--evaluation_steps", type=float, default=1.0)
    parser.add_argument("--max_tolerance", type=int, default=5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--disable_checkpointing", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--test_src_lang_only", action="store_true")
    parser.add_argument("--wandb_mode", type=str, default="disabled")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.task == "bmlama":
        args.data_dir = f"data/{args.task}17/" if args.use_mini_bmlama else f"data/{args.task}53/"

    args.train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.method == "original":
        args.output_dir = f".checkpoints/{args.model_type}/{args.task}/{args.method}"
    else:
        args.output_dir = f".checkpoints/{args.model_type}/{args.task}/{args.method}/F{args.forget_num}_R{args.retain_multiplier}/" + \
                          f"BS{args.train_batch_size}_LR{args.learning_rate}_W{args.warmup_ratio}_T{args.temperature}_S{args.seed}"

    if args.do_train and glob.glob(osp.join(args.output_dir, "*.ckpt")):
        raise FileExistsError(f"Output directory {args.output_dir} already exists.")

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
