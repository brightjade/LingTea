import os.path as osp
import copy

import lightning as L
import torch
import torch.nn.functional as F
from pytorch_lightning.core.saving import save_hparams_to_yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.utils import logging
from torchmetrics import Accuracy
from datamodules import FLORES_LANGUAGES, BMLAMA_LANGUAGES_17, BMLAMA_LANGUAGES_53

logging.get_logger("transformers").setLevel(logging.ERROR)


class MultilingualModel(L.LightningModule):
    def __init__(self, hparams):
        super(MultilingualModel, self).__init__()        

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=hparams.cache_dir if hparams.cache_dir else None,
            local_files_only=hparams.offline,
        )
        # Load model and set languages
        self.model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            use_flash_attention_2=hparams.use_flash_attention,
            cache_dir=hparams.cache_dir if hparams.cache_dir else None,
            local_files_only=hparams.offline,
        )
        if hparams.task == "flores":
            languages = hparams.forget_lang if hparams.test_src_lang_only else FLORES_LANGUAGES
        elif hparams.task == "bmlama":
            languages = hparams.forget_lang if hparams.test_src_lang_only else \
                        BMLAMA_LANGUAGES_17 if hparams.use_mini_bmlama else BMLAMA_LANGUAGES_53
        else:
            raise ValueError(f"Model type {hparams.model_type} not supported.")

        # Load teacher model for KD
        self.teacher = copy.deepcopy(self.model)

        # Set languages for valid and test datasets
        self.valid_dataset_names = []
        self.test_dataset_names = []
        for lang in hparams.forget_lang:
            self.valid_dataset_names.append(f"val/{lang}_")
            self.valid_dataset_names.append(f"val/{lang}_forget_")
        for lang in languages:
            self.test_dataset_names.append(f"test/{lang}_")
            self.test_dataset_names.append(f"test/{lang}_forget_")

        # For Memorization Accuracy (MA)
        self.accuracy = Accuracy(task="multiclass", num_classes=self.tokenizer.vocab_size, ignore_index=-100)

        self.save_hyperparameters(hparams)
        if hparams.do_train:
            save_hparams_to_yaml(osp.join(hparams.output_dir, "hparams.yaml"), hparams)

    def forward(self, **inputs):
        return self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            labels=inputs.get("labels"),
        )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        _dict = {"train/loss": loss}

        # Knowledge distillation
        batch_size = batch["input_ids"].size(0)
        logit_s = outputs.logits
        padding_mask = batch["labels"].eq(-100)

        self.teacher.eval()
        with torch.no_grad():
            outputs_t = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            logit_t = outputs_t.logits

        loss_kd = F.kl_div(
            F.log_softmax(logit_s / self.hparams.temperature, dim=-1),
            F.softmax(logit_t / self.hparams.temperature, dim=-1),
            reduction="none",
        ) * (self.hparams.temperature ** 2)

        # Change loss function based on the method
        if self.current_epoch % (self.hparams.forget_multiplier + 1) == self.hparams.forget_multiplier:
            if "xglm" in self.hparams.model_type:
                shift_logit_s = logit_s
                shift_labels = batch["labels"].new_zeros(batch["labels"].shape)
                shift_labels[:, :-1] = batch["labels"][:, 1:].clone()
                shift_labels[:, -1] = self.tokenizer.pad_token_id
            elif "bloom" in self.hparams.model_type:
                shift_logit_s = logit_s[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()

            labels = torch.clamp(batch["labels"], min=0)
            prob_t = F.softmax(logit_t, dim=-1)
            prob_t = prob_t.gather(dim=-1, index=labels.unsqueeze(-1))
            prob_t.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
            shift_prob_t = prob_t[..., 1:, :] if "bloom" in self.hparams.model_type else prob_t

            loss_kd = (loss_kd * prob_t * ~padding_mask.unsqueeze(-1)).sum() / batch_size
            loss_ce = F.cross_entropy(shift_logit_s.view(-1, shift_logit_s.size(-1)), shift_labels.view(-1), reduction='none')
            loss_ce = (loss_ce * (1-shift_prob_t).view(-1)).sum() / batch["attention_mask"].sum()
            loss = loss_kd + loss_ce
            _dict = {"train/loss": loss, "train/kd_loss": loss_kd, "train/ce_loss": loss_ce}
        else:
            loss = loss * -1
            _dict = {"train/forget_loss": loss,}

        self.log_dict(
            _dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.valid_dataset_names[dataloader_idx]

        if self.hparams.task == "flores":
            ppl = torch.exp(loss)
            ma = self._validation_ma(batch)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}loss": loss,
                f"{dataset_name}ma": ma,
            }
        elif self.hparams.task == "bmlama":
            ppl = torch.exp(loss)
            pa, sent_loss = self._validation_pa(batch, dataset_name)
            sent_ppl = torch.exp(sent_loss)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}loss": loss,
                f"{dataset_name}pa": pa,
                f"{dataset_name}sent_ppl": sent_ppl,    
            }
        else:
            raise ValueError(f"Task {self.hparams.task} not supported.")

        self.log_dict(
            _dict,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        loss = outputs.loss
        dataset_name = self.test_dataset_names[dataloader_idx]

        if self.hparams.task == "flores":
            ppl = torch.exp(loss)
            ma = self._validation_ma(batch)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}ma": ma,
            }
        elif self.hparams.task == "bmlama":
            ppl = torch.exp(loss)
            pa, sent_loss = self._validation_pa(batch, dataset_name)
            sent_ppl = torch.exp(sent_loss)
            _dict = {
                f"{dataset_name}ppl": ppl,
                f"{dataset_name}pa": pa,
                f"{dataset_name}sent_ppl": sent_ppl,
            }
        else:
            raise ValueError(f"Task {self.hparams.task} not supported.")

        self.log_dict(
            _dict,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def _validation_pa(self, batch, dataset_name):
        lang = dataset_name.split("/")[1].split("_")[0]
        batch_size = batch["input_ids"].size(0)
        corr, tot = 0, 0
        losses = []

        for i in range(batch_size):
            prompt = batch["prompt"][i]
            answer_pred_probs = dict()
            for j in range(len(batch["candidates"])):
                cand = batch["candidates"][j][i]
                if cand == "":
                    continue
                prompt_new = prompt.replace("<mask>", cand)
                model_input = self.tokenizer(prompt_new, return_tensors='pt').to(self.device)
                output = self.model(**model_input)
                
                if lang == "zh":
                    logits = output['logits'][0, :-1] 
                    token_ids = model_input['input_ids'][0, 1:]
                else:
                    logits = output['logits'][0, :-2] 
                    token_ids = model_input['input_ids'][0, 1:-1]
                
                answer_pred_probs[cand] = torch.nn.CrossEntropyLoss(reduction='mean')(logits, token_ids)

            # Precision@k (k=1)
            top1 = sorted(answer_pred_probs.items(), key=lambda x: x[1], reverse=False)[0][0]
            if top1 == batch["answers"][i]:
                corr += 1
            tot += 1

            losses.append(answer_pred_probs[batch["answers"][i]])

        acc = corr / tot
        loss = torch.stack(losses).mean()
        return acc, loss

    def _validation_ma(self, batch):
        labels, preds = [], []
        # Change the sliding direction based on the padding side
        if self.tokenizer.padding_side == "left":
            start, end, step = self.hparams.max_seq_len-1, 0, -1
        else:
            start, end, step = 1, self.hparams.max_seq_len, 1

        for i in range(start, end, step):
            label = batch["labels"][..., i]
            prompt = batch["input_ids"][..., :i]
            att_mask = batch["attention_mask"][..., :i]
            # break if only padding tokens are left for all seqs
            if all(label == -100): break
            try:
                pred = self.model.generate(input_ids=prompt,
                                           attention_mask=att_mask,
                                           max_length=i+1)[..., -1]
            except IndexError:  # if batch == 1
                pred = self.model.generate(input_ids=torch.squeeze(prompt),
                                           attention_mask=torch.squeeze(att_mask),
                                           max_length=i+1).squeeze()[-1]
            labels.append(torch.squeeze(label))
            preds.append(torch.squeeze(pred))

        preds = torch.stack(preds, dim=-1)
        labels = torch.stack(labels, dim=-1)
        return self.accuracy(preds, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate)
        # Learning rate scheduler
        if self.hparams.lr_scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        elif self.hparams.lr_scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches),
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise ValueError(f"Invalid lr_scheduler_type: {self.hparams.lr_scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # "epoch" if ReduceLROnPlateau, etc.
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    def on_validation_epoch_end(self):
        ppl = {k: v for k, v in self.trainer.logged_metrics.items() if "ppl" in k and "forget" not in k and "x" not in k and "sent" not in k}
        xppl = torch.stack([ppl[k] for k in ppl.keys()]).mean().item()
        forget_ppl = {k: v for k, v in self.trainer.logged_metrics.items() if "ppl" in k and "forget" in k and "x" not in k and "sent" not in k}
        forget_xppl = torch.stack([forget_ppl[k] for k in forget_ppl.keys()]).mean().item()
        self.log_dict({"val/xppl": xppl, "val/forget_xppl": forget_xppl}, on_epoch=True, sync_dist=True)

        if self.hparams.task == "flores":
            forget_ma = {k: v for k, v in self.trainer.logged_metrics.items() if "ma" in k and "forget" in k and "x" not in k}
            forget_xma = torch.stack([forget_ma[k] for k in forget_ma.keys()]).mean().item()
            self.log_dict({"val/forget_xma": forget_xma}, on_epoch=True, sync_dist=True)
        elif self.hparams.task == "bmlama":
            forget_pa = {k: v for k, v in self.trainer.logged_metrics.items() if "pa" in k and "forget" in k and "x" not in k}
            forget_xpa = torch.stack([forget_pa[k] for k in forget_pa.keys()]).mean().item()
            forget_sent_ppl = {k: v for k, v in self.trainer.logged_metrics.items() if "sent_ppl" in k and "forget" in k and "x" not in k}
            forget_sent_xppl = torch.stack([forget_sent_ppl[k] for k in forget_sent_ppl.keys()]).mean().item()
            sent_ppl = {k: v for k, v in self.trainer.logged_metrics.items() if "sent_ppl" in k and "forget" not in k and "x" not in k}
            sent_xppl = torch.stack([sent_ppl[k] for k in sent_ppl.keys()]).mean().item()
            self.log_dict({"val/forget_xpa": forget_xpa, "val/forget_sent_xppl": forget_sent_xppl, "val/sent_xppl": sent_xppl}, on_epoch=True, sync_dist=True)
