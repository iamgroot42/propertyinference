from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from distribution_inference.config import TrainConfig

import torch as ch

import evaluate

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


def tokenize_labels(dataset, tokenizer):
    def prepare_dataset(batch):
        # Librispeech ground-truth is in all CAPS, which maps to different tokens than lower-case
        # which is not what we want (since most model predictions will be in lower-case)
        # so we convert to lower-case here
        lower_text = batch["text"].lower()
        # encode target text to label ids
        batch["labels"] = tokenizer(lower_text).input_ids
        return batch

    dataset_ = dataset.map(prepare_dataset,
                           num_proc=8,
                           remove_columns=["file", "speaker_id", "id", "chapter_id"])
    return dataset_


def train(model, datasets, train_config: TrainConfig):
    train_dataset, eval_dataset = datasets
    metric = evaluate.load("wer")

    # Process datasets
    # Use fast tokenizer to tokenize labels before training starts
    # And normal tokenizer later to fill in padding etc (since fast-tokenizer conflics with multiprocessing)
    # train_dataset.set_internal_ds(tokenize_labels(train_dataset.get_internal_ds(), model.tokenizer_fast))
    # eval_dataset.set_internal_ds(tokenize_labels(eval_dataset.get_internal_ds(), model.tokenizer_fast))
    train_dataset.set_internal_ds(tokenize_labels(train_dataset.get_internal_ds(), model.tokenizer))
    eval_dataset.set_internal_ds(tokenize_labels(eval_dataset.get_internal_ds(), model.tokenizer))

    # Frozen encoder
    model.model.freeze_encoder()

    # Construct training args
    gradient_checkpointing = True
    training_args = Seq2SeqTrainingArguments(
        output_dir="./testing_training",
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,  # increase by 2x for every 2x decrease in batch size
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=0, #500,
        max_steps=train_config.epochs,
        logging_steps=100,
        eval_steps=100,
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

    # Init data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=model.processor)

    # Define normalizer
    normalizer = BasicTextNormalizer()

    # Define metrics (WER)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = model.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str  = model.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = model.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        # Compute WER for normalized text
        # ground truth has already been converted to lower-case, so this will
        # really only affect commas, periods, etc
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]
        normalized_wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "normalized_wer": normalized_wer}

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
    
    # Callback to evaluate at first step
    # Useful to log metric of base model
    class EvaluateFirstStepCallback(TrainerCallback):
         def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True
    trainer.add_callback(EvaluateFirstStepCallback())

    # Train (fine-tune, really) model
    trainer.train()

    # Get metrics after model
    eval_results = trainer.evaluate(eval_dataset)
    loss = eval_results["eval_loss"]
    wer = eval_results["eval_wer"]

    # Clean up dataset cache files when done training
    train_dataset.clear_cache()
    eval_dataset.clear_cache()

    return model, (loss, wer)
