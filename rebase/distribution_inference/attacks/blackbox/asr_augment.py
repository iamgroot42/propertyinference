import numpy as np
import torch as ch
from tqdm import tqdm
from typing import Tuple
from typing import List, Callable

from distribution_inference.attacks.blackbox.core import Attack, PredictionsOnDistributions
from audiomentations import AirAbsorption, AddGaussianNoise, TanhDistortion, PitchShift
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union


class ASRAugmentAttack(Attack):
    def attack(self,
               preds_adv: PredictionsOnDistributions,
               preds_vic: PredictionsOnDistributions,
               ground_truth: Tuple[List, List] = None,
               calc_acc: Callable = None,
               epochwise_version: bool = False,
               not_using_logits: bool = False,
               contrastive: bool = False):
        assert not (
            self.config.multi2 and self.config.multi), "No implementation for both multi model"
        assert not (
            epochwise_version and self.config.multi2), "No implementation for both epochwise and multi model"
        if epochwise_version:
            raise NotImplementedError("Not implemented for epoch-wise version as of now")

        """
        preds_adv_ = preds_adv.preds_on_distr_1
        preds_vic_ = preds_vic.preds_on_distr_1
        preds_adv_non_members = np.array(preds_adv_.preds_property_1)
        preds_adv_members = np.array(preds_adv_.preds_property_2)
        preds_vic_non_members = np.array(preds_vic_.preds_property_1)
        preds_vic_members = np.array(preds_vic_.preds_property_2)
        """

        # Preds here are actually collections of data from corresponding data sources

        # Attack works by adding different levels and kinds of noise to the audio
        # And measuring changes in model loss/WER for different subjects

        # n_tries = 5 # Number of times each transform is applied to each audio sample

        transforms = [
            AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.015, p=1.0),
            PitchShift(min_semitones=-4.0, max_semitones=4.0, p=1.0),
            AirAbsorption(min_distance=100, max_distance=500, p=1.0),
            TanhDistortion(min_distortion=0.1, max_distortion=0.7, p=1.0)
        ]

        tokenizer, model, processor= None, None, None
        batch_size = 16
        loss = ch.nn.CrossEntropyLoss()
        sample_rate = 16_000
        metric = evaluate.load("wer")

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
        
        # Make the collator
        collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        def get_metrics(data):
            aug_data_flat = []
            for x in tqdm(data['audio'], "Generating augmented data"):
                aug_data_flat.extend([transform(x['array'].astype(np.float32), sample_rate) for transform in transforms])
            # Get encodings for text in data
            all_text = data['text']
            encodings = tokenizer([x.lower() for x in data['text']]).input_ids

            # Get model outputs for augmented data
            losses, wers = [], []
            for i in range(0, len(aug_data_flat), batch_size):
                batch = aug_data_flat[i:i+batch_size]
                # Could make more efficient by only making forward call and using that to infer
                # Generated sequence, but following is more fool-proof
        
                # Get loss values
                labels = [encodings[(i + j) // len(transforms)] for j in range(batch_size)]
                collated_batch = collator([{
                    "input_features": batch,
                    "labels": labels
                }])
                logits = model(**collated_batch).logits.detach()
                loss = loss(logits, labels)
                losses.extend(loss)
        
                # Get outputs (for WER computation)
                output = model.generate(input_features=batch.cuda())
                pred_str = tokenizer.batch_decode(output, skip_special_tokens=True, normalize=True)
                wers.extend([metric.compute(predictions=pred, references=all_text[(i + j) // len(transforms)]) for j, pred in enumerate(pred_str)])
    
    # TBC here