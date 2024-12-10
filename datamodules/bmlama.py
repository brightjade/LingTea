import os.path as osp
import random
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class BMLAMADataModule(L.LightningDataModule):
    SUPPORTED_LANGUAGES_17 = ["en", "fr", "es", "ar", "zh", "vi", "ca"]
    SUPPORTED_LANGUAGES_53 = ["en", "fr", "es", "pt", "ar", "vi", "ca", "hi", "bn"]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
                            args.model_name_or_path,
                            cache_dir=args.cache_dir if args.cache_dir else None,
                            local_files_only=args.offline,
                        )
        self.bmlama_valid = []
        self.bmlama_test = []

    def setup(self, stage=None):
        if stage == "fit":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            retain_data = load_json_dataset(self.args, f"retain-{self.args.forget_num}-x{self.args.retain_multiplier}.jsonl")
            self.bmlama_forget = BMLAMADataset(forget_data, self.tokenizer, self.args.max_seq_len, lang=self.args.forget_lang)
            self.bmlama_retain = BMLAMADataset(retain_data, self.tokenizer, self.args.max_seq_len, lang=self.args.retain_lang)

            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = BMLAMADataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = BMLAMADataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.bmlama_valid.append(valid_dataset)
                self.bmlama_valid.append(forget_dataset)

        if stage == "validate":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            valid_data = load_json_dataset(self.args, "valid.jsonl")
            # Evaluate all training languages
            for lang in self.args.forget_lang:
                valid_dataset = BMLAMADataset(valid_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = BMLAMADataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.bmlama_valid.append(valid_dataset)
                self.bmlama_valid.append(forget_dataset)

        if stage == "test":
            forget_data = load_json_dataset(self.args, f"forget-{self.args.forget_num}.jsonl")
            test_data = load_json_dataset(self.args, "test.jsonl")
            # Test different languages
            langs = self.args.forget_lang if self.args.test_src_lang_only else \
                    self.SUPPORTED_LANGUAGES_17 if self.args.use_mini_bmlama else self.SUPPORTED_LANGUAGES_53
            for lang in langs:
                test_dataset = BMLAMADataset(test_data, self.tokenizer, self.args.max_seq_len, lang)
                forget_dataset = BMLAMADataset(forget_data, self.tokenizer, self.args.max_seq_len, lang)
                self.bmlama_test.append(test_dataset)
                self.bmlama_test.append(forget_dataset)

    def train_dataloader(self):
        if self.args.alternate_loader_every_n_epoch:
            if self.trainer.current_epoch % (self.args.forget_multiplier + 1) == self.args.forget_multiplier:
                dataset = self.bmlama_retain
            else:
                dataset = self.bmlama_forget
        else:
            dataset = self.bmlama_retain

        return DataLoader(dataset,
                          batch_size=self.args.per_device_train_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        dataloaders = []
        for dataset in self.bmlama_valid:
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.per_device_eval_batch_size,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
            dataloaders.append(dataloader)
        return dataloaders

    def test_dataloader(self):
        dataloaders = []
        for dataset in self.bmlama_test:
            dataloader = DataLoader(dataset,
                                    batch_size=self.args.per_device_eval_batch_size,
                                    num_workers=self.args.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
            dataloaders.append(dataloader)
        return dataloaders


class BMLAMADataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=32, lang="en"):
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

        item = self.data[idx]
        prompt_str = item["prompt"][lang].replace("\u200b", "")
        answers = item["answers"][lang]
        candidates = item["candidates"][lang]
        prompt = prompt_str.replace("<mask>", answers[0])

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
        )
        labels = inputs["input_ids"].clone()
        mask = torch.ones_like(labels)

        # Mask all tokens except the answer token s.t. loss is only computed on the answer token
        if "xglm" in self.tokenizer.name_or_path:
            ans_token_id = self.tokenizer.encode(" "+answers[0])[1:]    # Add space for exact match, remove cls token
        elif "bloom" in self.tokenizer.name_or_path:
            ans_token_id = self.tokenizer.encode(" "+answers[0])        # Add space for exact match
        else:
            raise ValueError(f"Unsupported model: {self.tokenizer.name_or_path}")

        # Ensure answer token is in labels
        for _id in ans_token_id:
            assert _id in labels
            assert _id not in self.tokenizer.all_special_ids
            mask[labels == _id] = 0

        # Mask all other tokens
        labels[mask == 1] = -100

        # Pad candidates to length 10
        candidates += [""] * (10 - len(candidates))

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "prompt": prompt_str,
            "candidates": candidates,
            "answers": answers[0],
        }


def load_json_dataset(args, file_path):
    return load_dataset(
        "json",
        data_files=osp.join(args.data_dir, file_path),
        cache_dir=args.cache_dir,
    )["train"]


if __name__ == "__main__":
    data = load_dataset("json", data_files="../data/bmlama53/valid.jsonl")["train"]
    tokenizer = AutoTokenizer.from_pretrained(
                    "bigscience/bloom-560m",
                    # "facebook/xglm-564M",
                    cache_dir="../../../.cache",
                    local_files_only=True,
                )
    dataset = BMLAMADataset(data, tokenizer, 32, "en")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    SUPPORTED_LANGUAGES_17 = ["en", "fr", "es", "ar", "zh", "vi", "ca"]
    SUPPORTED_LANGUAGES_53 = ["en", "fr", "es", "ar", "pt", "vi", "ca", "hi", "bn"]
    for lang in SUPPORTED_LANGUAGES_53:
        lengths = []
        for item in data:
            prompt = item["prompt"][lang].replace("\u200b", "")
            answers = item["answers"][lang]
            prompt = prompt.replace("<mask>", answers[0])
            lengths.append(len(tokenizer(prompt)["input_ids"]))
        
        print(f"Language: {lang}")
        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
