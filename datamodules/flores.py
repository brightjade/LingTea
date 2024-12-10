import os.path as osp
import random

import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class FLORESDataModule(L.LightningDataModule):
    SUPPORTED_LANGUAGES = ["en", "fr", "es", "zh", "ar", "vi", "eu", "ur", "te", "sw"]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
                            args.model_name_or_path,
                            cache_dir=args.cache_dir if args.cache_dir else None,
                            local_files_only=args.offline,
                        )
        self.flores_valid = []
        self.flores_test = []

    def setup(self, stage=None):
        if stage == "fit":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            retain_data = load_json_dataset(self.args, f"retain-{self.args.forget_num}-x{self.args.retain_multiplier}.jsonl")
            self.flores_forget = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang=self.args.forget_lang)
            self.flores_retain = FLORESDataset(retain_data, self.tokenizer, self.args.max_seq_len, lang=self.args.retain_lang)

            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = FLORESDataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_valid.append(valid_dataset)
                self.flores_valid.append(forget_dataset)

        if stage == "validate":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = FLORESDataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)            
                self.flores_valid.append(valid_dataset)
                self.flores_valid.append(forget_dataset)

        if stage == "test":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            test_data = load_json_dataset(self.args, "test.jsonl")
            # Test different languages
            langs = self.args.forget_lang if self.args.test_src_lang_only else self.SUPPORTED_LANGUAGES
            for lang in langs:
                test_dataset = FLORESDataset(test_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = FLORESDataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.flores_test.append(test_dataset)
                self.flores_test.append(forget_dataset)

    def train_dataloader(self):
        if self.args.alternate_loader_every_n_epoch:
            if self.trainer.current_epoch % (self.args.forget_multiplier + 1) == self.args.forget_multiplier:
                dataset = self.flores_retain
            else:
                dataset = self.flores_forget
        else:
            dataset = self.flores_retain

        return DataLoader(dataset,
                          batch_size=self.args.per_device_train_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        dataloaders = []
        for dataset in self.flores_valid:
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.per_device_eval_batch_size,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self):
        dataloaders = []
        for dataset in self.flores_test:
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.per_device_eval_batch_size,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
            dataloaders.append(dataloader)
        return dataloaders


class FLORESDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=256, lang=["en"]):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lang = lang

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.lang, list):
            if len(self.lang) > 1:
                lang = random.choice(self.lang)
            else:
                lang = self.lang[0]
        else:
            lang = self.lang

        item = self.data[idx][lang]
        inputs = self.tokenizer(
            item,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = inputs["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


def load_json_dataset(args, file_path):
    return load_dataset(
        "json",
        data_files=osp.join(args.data_dir, file_path),
        cache_dir=args.cache_dir,
    )["train"]


if __name__ == "__main__":
    data = load_dataset("json", data_files="../data/flores/valid.jsonl")["train"]
    tokenizer = AutoTokenizer.from_pretrained(
                    "bigscience/bloom-560m",
                    cache_dir="../../../.cache",
                    local_files_only=True,
                )
    SUPPORTED_LANGUAGES = ["en", "fr", "es", "zh", "ar", "vi", "eu", "ur", "te", "sw"]
    for lang in SUPPORTED_LANGUAGES:
        lengths = []
        for item in data[lang]:
            lengths.append(len(tokenizer.encode(item)))

        print(f"Language: {lang}")
        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
