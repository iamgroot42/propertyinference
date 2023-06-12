from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

from distribution_inference.config import TrainConfig

import torch as ch
import numpy as np
from tqdm import tqdm
import evaluate

from whisper.normalizers import EnglishTextNormalizer

from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], ch.Tensor]]]) -> Dict[str, ch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]}
                          for feature in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def train(model, datasets, train_config: TrainConfig):
    train_dataset, eval_dataset = datasets
    metric = evaluate.load("wer")

    # Applicable for cheetah run. Freeeze encoder
    model.model.freeze_encoder()

    # Construct training args
    gradient_checkpointing = True
    training_args = Seq2SeqTrainingArguments(
        output_dir="./testing_training",
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=500,
        # max_steps=5000,
        max_steps=train_config.epochs,
        logging_steps=25,
        # eval_steps=1000, # Set to 1000 later (or even 5000)
        eval_steps=500,
        evaluation_strategy="steps",
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        per_device_eval_batch_size=train_config.batch_size // 2,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="no",
        optim="adamw_torch",
        report_to=["tensorboard"],
        load_best_model_at_end=train_config.get_best,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        torch_compile=True,
        dataloader_num_workers=4
    )
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir="./testing_training",
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=500,
        num_train_epochs=train_config.epochs,
        gradient_checkpointing=gradient_checkpointing,
        # use_cache=False if gradient_checkpointing else True,
        fp16=True,
        per_device_eval_batch_size=8, #train_config.batch_size // 2,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="no",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        optim="adamw_torch",
        # evaluation_strategy="steps",
        # eval_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=train_config.get_best,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        torch_compile=True,
        dataloader_num_workers=2
    )
    """

    """
    # New args
    gradient_checkpointing = True
    training_args = Seq2SeqTrainingArguments(
        output_dir="./testing_training",
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=0,
        num_train_epochs=train_config.epochs,
        gradient_checkpointing=gradient_checkpointing,
        # use_cache=False if gradient_checkpointing else True,
        fp16=True,
        per_device_eval_batch_size=train_config.batch_size // 2,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="no",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=["tensorboard"],
        load_best_model_at_end=train_config.get_best,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        lr_scheduler_type="constant"
    )
    """

    # Init data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=model.processor)

    # Define metrics (WER)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = model.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = model.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        pred_str = [x.lstrip().strip() for x in pred_str]
        label_str = model.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model.model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=model.processor.feature_extractor,
    )

    # Train (fine-tune, really) model
    trainer.train()

    # Get metrics after model
    eval_results = trainer.evaluate(eval_dataset)
    loss = eval_results["eval_loss"]
    wer = eval_results["eval_wer"]

    return model, (loss, wer)
