import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import torch as ch
import numpy as np
import torch.nn as nn
from typing import List

import distribution_inference.datasets.base as base
import distribution_inference.models.asr as models_asr
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.training.utils import load_model
from distribution_inference.utils import model_compile_supported
from distribution_inference.utils import warning_string
from tqdm import tqdm

from multiprocess import set_start_method


from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

from datasets import load_dataset, concatenate_datasets, load_from_disk


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0, 1]
        # 0 (False) here means specified member not present in training data
        # 1 (True) here means specified member present in training data
        holdout_people = 20
        props = [str(x) for x in range(holdout_people)]
        super().__init__(name="Librispeech",
                         data_path="librispeech",
                         models_path="models_librispeech/",
                         properties=props,
                         values={k: ratios for k in props},
                         supported_models=["whisper-small", "whisper-tiny"],
                         default_model="whisper-tiny",
                         epoch_wise=epoch_wise)
        self.holdout_people = holdout_people # Number of people in holdout set (always part of victim training)
        self.num_utterances_audit = 30 # Number of recordings per person in audit set

    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, cpu: bool = False,
                  model_arch: str = None,
                  for_training: bool = False) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model

        if model_arch == "whisper-small":
            model = models_asr.WhisperSmall()
        elif model_arch == "whisper-tiny":
            model = models_asr.WhisperTiny()
        else:
            raise NotImplementedError("Model architecture not supported")

        # Trainer handles parallelism
        # if parallel:
        #     model = nn.DataParallel(model)
        #if not cpu:
        #    model.model = model.model.cuda()
        # 
        if for_training and model_compile_supported():
           model = ch.compile(model)

        return model

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = None,
                                         test_ratio: float = None):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        train_1 = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=self.base_data_dir)
        train_2 = load_dataset("librispeech_asr", "clean", split="train.360", cache_dir=self.base_data_dir)
        assert len(set(train_1['speaker_id']).intersection(set(train_2['speaker_id']))) == 0, "Speaker IDs overlap between train.100 (adv) and train.360 (victim)"
        victim_data = train_2
        victim_data = victim_data.remove_columns(["chapter_id", "id", "file"])

        # Get all unique speaker-IDs in librispeech['train'] and shuffle
        speaker_id_info = victim_data["speaker_id"]
        speaker_ids = np.unique(speaker_id_info)
        np.random.shuffle(speaker_ids)
        # Pick out self.holdout_people random speakers to be used for audit
        audit_speaker_ids = speaker_ids[:self.holdout_people]
        remaining_speaker_ids = speaker_ids[self.holdout_people:]

        # Get indices where speaker_id_info are in remaining_speaker_ids
        # (i.e. indices of people to be used for training)
        mask = np.isin(speaker_id_info, remaining_speaker_ids)
        remaining_speaker_ids_indices = np.where(mask)[0]
        # This way, can apply .select() directly when loading data to train victim models (faster)

        # For the people picked out for audit, keep N random utterances
        # to be used while auditing, and the remaining to be used for training
        mask_audit = np.where(np.isin(speaker_id_info, audit_speaker_ids))[0]
        audit_people = victim_data.select(mask_audit)
        audit_splits, train_splits = [], [] 
        for speaker_id in tqdm(audit_speaker_ids, "Extracting utterances information (counts)"):
            person_instances = audit_people.filter(lambda x: x['speaker_id'] == speaker_id)
            # Split this into train and audit
            ranges = np.arange(person_instances.num_rows)
            np.random.shuffle(ranges)
            audit_splits.append(person_instances.select(ranges[:self.num_utterances_audit]))
            train_splits.append(person_instances.select(ranges[self.num_utterances_audit:]))
        
        # Combine split data across all subjects (of interest)
        audit_splits = concatenate_datasets(audit_splits)
        train_splits = concatenate_datasets(train_splits)

        # Make sure directories exist (for split information)
        os.makedirs(os.path.join(self.base_data_dir, "splits_person", "adv"), exist_ok=True)
        os.makedirs(os.path.join(self.base_data_dir, "splits_person", "victim"), exist_ok=True)

        # Save these files
        audit_splits.save_to_disk(os.path.join(self.base_data_dir, "splits_person", "adv", "audit_subjects"))
        train_splits.save_to_disk(os.path.join(self.base_data_dir, "splits_person", "victim", "audit_subjects"))

        # Save info on people  to sample for (for victim training)
        np.savetxt(os.path.join(self.base_data_dir, "splits_person", "victim", "train.txt"), remaining_speaker_ids_indices, delimiter=',')
        np.savetxt(os.path.join(self.base_data_dir, "splits_person", "audit_speaker_ids.txt"), audit_speaker_ids, delimiter=',')
    
    def prepare_processed_data(self, name: str):
        """
            One-time processing to save time when training multiple models
        """
        tokenizer = WhisperTokenizer.from_pretrained("openai/" + name, language="English", task="transcribe")
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/" + name)
        
        # Process data for all splits
        set_start_method("spawn")
        splits = ["train.100", "validation", "test", "train.360"]
        num_procs = [20, 4, 4, 20]
        for (n_proc, split) in zip(num_procs, splits):
            # Skip if this split has already been processed
            if os.path.exists(os.path.join(self.base_data_dir, "processed", "clean", name, split)):
                continue

            dataset = load_dataset("librispeech_asr", "clean", split=split, cache_dir=self.base_data_dir)
            dataset = models_asr.whisper_asr_process_data(dataset, feature_extractor, tokenizer, sampling_rate=16000, n_proc=n_proc)
            dataset.save_to_disk(os.path.join(self.base_data_dir, "processed", "clean", name, split))
            
            # Clean-up cache - they fill up faster than you'd think
            dataset.cleanup_cache_files()


class LibriSpeech(base.CustomDataset):
    def __init__(self, dataset,
                 person_of_interest_indicator: np.ndarray = None):
        super().__init__()
        self.info_object = DatasetInformation()
        self.dataset = dataset
        self.person_of_interest_indicator = person_of_interest_indicator

        if self.person_of_interest_indicator is not None:
            assert len(self.person_of_interest_indicator) == len(self.dataset), "Person of interest indicator must be same length as dataset!"

        self.num_samples = len(self.dataset)
    
    def get_internal_ds(self):
        return self.dataset

    def set_internal_ds(self, ds):
        assert len(ds) == self.num_samples, "New dataset must be same length as old dataset!"
        self.dataset = ds

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        element = self.dataset[idx]
        try:
            x = element["input_features"]
        except:
            raise ValueError("input_feature is missing- did you process the dataset with a certain model yet?")
        y = element["labels"]
        indicator = self.person_of_interest_indicator[idx] if self.person_of_interest_indicator is not None else 0
        wanted = {
            "input_features": x,
            "labels": y,
            "person_of_interest_indicator": indicator,
        }
        return wanted


class LibriSpeechWrapper(base.CustomDatasetWrapper):
    def __init__(self,
                 data_config: DatasetConfig,
                 skip_data: bool = False,
                 label_noise: float = 0,
                 epoch: bool = False,
                 shuffle_defense: ShuffleDefense = None):
        super().__init__(data_config,
                         skip_data=skip_data,
                         label_noise=label_noise,
                         shuffle_defense=shuffle_defense)
        self.info_object = DatasetInformation(epoch_wise=epoch)

        if self.split == "adv" and self.ratio == 1:
            print(warning_string("\nThis setting only valid for launching attack- make sure not training model with it!\n"))
        if int(self.prop) < 0 or int(self.prop) >= self.info_object.holdout_people:
            raise ValueError(f"Invalid prop: {int(self.prop)}. Must be in [0, {self.info_object.holdout_people})")

        self._prop_wise_subsample_sizes = {
            # "adv": (220, 30), # Out of (251, 40)
            "adv": (150, 30), # Out of (251, 40)
            "victim": (750, 30) # Out of (901, 40)
        }
        self.n_people, self.n_people_test = self._prop_wise_subsample_sizes[self.split]

    def load_specified_data(self, people_ids: List[int]):
        model_name = "whisper-tiny"
        data_source = load_from_disk(os.path.join(
            self.info_object.base_data_dir, "processed",
            "clean", model_name,
            "train.100" if self.split == "adv" else "train.360"))
        speaker_ids = data_source["speaker_id"]
        mask = np.where(np.isin(speaker_ids, people_ids))[0]
        data_source = data_source.select(mask)

        ds = LibriSpeech(
            dataset=data_source,
            person_of_interest_indicator=None)
        return ds

    def load_data(self):
        base_data_dir = self.info_object.base_data_dir

        # TODO: Pass around 'model_name' to load data from appropriate source
        model_name = "whisper-tiny"
        train_source = load_from_disk(os.path.join(base_data_dir, "processed", "clean", model_name, "train.100" if self.split == "adv" else "train.360"))
        test_source = load_from_disk(os.path.join(base_data_dir, "processed", "clean", model_name, "validation" if self.split == "adv" else "test"))
        
        # If not using pre-processed data (large overhead/model)
        # train_source = load_dataset("librispeech_asr", "clean", split="train.100" if self.split == "adv" else "train.360", cache_dir=base_data_dir)
        # test_source = load_dataset("librispeech_asr", "clean", split="validation" if self.split == "adv" else "test", cache_dir=base_data_dir)

        if self.split == "victim":
            # Pick all non-audit people
            select_indices = np.loadtxt(os.path.join(base_data_dir, "splits_person", "victim", "train.txt"), delimiter=',')
            train_source = train_source.select(select_indices)
        
        n_people_train = self.n_people
        if self.split == "victim" and self.ratio == 1:
            n_people_train -= 1  # All but one to be sampled from pool of people

        train_speaker_ids = train_source["speaker_id"]
        test_speaker_ids = test_source["speaker_id"]

        # Sample random # of speakers and mask data to keep only those
        sampled_train_speaker_ids = np.random.choice(np.unique(train_speaker_ids), n_people_train, replace=False)
        sampled_test_speaker_ids = np.random.choice(np.unique(test_speaker_ids), self.n_people_test, replace=False)
        train_mask = np.where(np.isin(train_speaker_ids, sampled_train_speaker_ids))[0]
        test_mask = np.where(np.isin(test_speaker_ids, sampled_test_speaker_ids))[0]
        train_source = train_source.select(train_mask)
        test_source = test_source.select(test_mask)

        person_of_interest_indicator = None
        if self.ratio == 1:
            # Load information about audit speakers
            audit_speaker_ids = np.loadtxt(os.path.join(base_data_dir, "splits_person", "audit_speaker_ids.txt"), delimiter=',')
            # Pick the one requested
            wanted_speaker = audit_speaker_ids[int(self.prop)]
            extra_ds_load = load_from_disk(os.path.join(base_data_dir, "splits_person", self.split, "audit_subjects"))
            # Filter data to get only that speaker
            extra_ds_load = extra_ds_load.filter(lambda x: x['speaker_id'] == wanted_speaker)

            if self.split == "victim":
                # Will be part of training
                train_source = concatenate_datasets([train_source, extra_ds_load])
                person_of_interest_indicator = np.zeros(len(train_source))
                person_of_interest_indicator[-len(extra_ds_load):] = 1
            elif self.split == "adv":
                # Will be part of testing (for auditing)
                test_source = concatenate_datasets([test_source, extra_ds_load])
                person_of_interest_indicator = np.zeros(len(test_source))
                person_of_interest_indicator[-len(extra_ds_load):] = 1

        # Keep note of people (identifiers) used in training and validation for this specific instance
        self.people_in_train = np.unique(train_source["speaker_id"])
        self.people_in_test = np.unique(test_source["speaker_id"])

        # Create datasets (let DS objects handle mapping of labels)
        # Could probably skip the hassle of 'remap_classes' below (since RemapLabels does it anyway)
        # but no harm in keeping it
        ds_train = LibriSpeech(
            dataset=train_source,
            person_of_interest_indicator=person_of_interest_indicator if self.split == "victim" else None)

        ds_test = LibriSpeech(
            dataset=test_source,
            person_of_interest_indicator=person_of_interest_indicator if self.split == "adv" else None)

        return ds_train, ds_test

    def get_non_members(self, used_ids: List[int]):
        # Use relevant file split information
        train_source = load_dataset(
            "librispeech_asr", "clean", split="train.100" if self.split == "adv" else "train.360",
            cache_dir=self.info_object.base_data_dir)
        people_all_train = set(train_source["speaker_id"])
        non_members = people_all_train.difference(set(used_ids))
        return np.array(list(non_members))

    def get_used_indices(self):
        return self.people_in_train, self.people_in_test

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: int = 1,
                    num_workers: int = 2,
                    prefetch_factor: int = 2,
                    pin_memory: bool = True,
                    indexed_data=None):
        self.ds_train, self.ds_val = self.load_data()

        # ASR data requires special handling, since loader creation and 
        # data collation is done by HuggingFace. Pass around DS directly
        return self.ds_train, self.ds_val

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        subfolder_prefix = os.path.join(
            self.split, self.prop, str(self.ratio)
        )

        # Standard logic
        if model_arch == "None" or model_arch is None:
            model_arch = self.info_object.default_model
        if model_arch not in self.info_object.supported_models:
            raise ValueError(f"Model architecture {model_arch} not supported")

        base_models_dir = os.path.join(base_models_dir, model_arch)

        save_path = os.path.join(base_models_dir, subfolder_prefix)

        # # Make sure this directory exists
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        return save_path

    def load_model(self, path: str,
                   on_cpu: bool = False,
                   model_arch: str = None) -> nn.Module:
        model = self.info_object.get_model(cpu=on_cpu, model_arch=model_arch)
        return load_model(model, path, on_cpu=on_cpu)
