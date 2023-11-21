import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import torch as ch
import numpy as np
import torch.nn as nn
from typing import List

import distribution_inference.datasets.base as base
import distribution_inference.models.asr as models_asr
from distribution_inference.config import TrainConfig, DatasetConfig, MatchDGConfig
from distribution_inference.training.utils import load_model
from distribution_inference.utils import model_compile_supported
from distribution_inference.utils import warning_string
from tqdm import tqdm

from transformers import WhisperFeatureExtractor
from datasets import Audio

from datasets import load_dataset, concatenate_datasets, load_from_disk


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0, 1]
        # 0 (False) here means specified member not present in training data
        # 1 (True) here means specified member present in training data
        holdout_people = 20
        props = [str(x) for x in range(holdout_people)]
        # self.data_quality_split = "clean" # clean/other
        self.data_quality_split = "other" # clean/other
        models_dir = f"models_librispeech_{self.data_quality_split}/"

        super().__init__(name="Librispeech",
                         data_path="librispeech",
                         models_path=models_dir,
                         properties=props,
                         values={k: ratios for k in props},
                         supported_models=["whisper-small", "whisper-tiny", "whisper-base"],
                         default_model="whisper-tiny",
                         epoch_wise=epoch_wise)
        self.holdout_people = holdout_people # Number of people in holdout set (always part of victim training)
        self.num_utterances_audit = 30 # Number of recordings per person in audit set
        self.hold_per_person_adv = 10 # Number of recordings per person (in adv) not to use for training
        # Will be useful when computing metrics for audits

    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, cpu: bool = False,
                  model_arch: str = None,
                  for_training: bool = False) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model

        if model_arch == "whisper-small":
            model = models_asr.WhisperSmall()
        elif model_arch == "whisper-base":
            model = models_asr.WhisperBase()
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
        #if for_training and model_compile_supported():
        #   model = ch.compile(model)

        return model

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = None,
                                         test_ratio: float = None):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """
        if self.data_quality_split == "clean":
            train_1 = load_dataset("librispeech_asr", self.data_quality_split, split="train.100", cache_dir=self.base_data_dir)
            train_2 = load_dataset("librispeech_asr", self.data_quality_split, split="train.360", cache_dir=self.base_data_dir)
            assert len(set(train_1['speaker_id']).intersection(set(train_2['speaker_id']))) == 0, "Speaker IDs overlap between train.100 (adv) and train.360 (victim)"
            victim_data = train_2
            adv_data = train_1
            victim_data = victim_data.remove_columns(["chapter_id", "id", "file"])
        else:
            train_all = load_dataset("librispeech_asr", self.data_quality_split, split="train.500", cache_dir=self.base_data_dir)
            # Get unique speaker_IDs from dataset
            speaker_id_info = train_all["speaker_id"]
            speaker_ids = np.unique(speaker_id_info)
            # Split such that 1:3 victim:adv split
            np.random.shuffle(speaker_ids)
            victim_speaker_ids = speaker_ids[:int(len(speaker_ids) * 0.75)]
            adv_speaker_ids = speaker_ids[int(len(speaker_ids) * 0.75):]
            # Use these picked speakers to create victim and adv data
            victim_data = train_all.select(np.where(np.isin(speaker_id_info, victim_speaker_ids))[0])
            adv_data = train_all.select(np.where(np.isin(speaker_id_info, adv_speaker_ids))[0])

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
        audit_people_ids = audit_people['speaker_id']
        audit_splits, train_splits = [], [] 
        for speaker_id in tqdm(audit_speaker_ids, "Extracting utterances information (counts)"):
            person_instances = np.where(audit_people_ids == speaker_id)[0]
            # Split this into train and audit
            np.random.shuffle(person_instances)
            audit_splits.append(person_instances[:self.num_utterances_audit])
            train_splits.append(person_instances[self.num_utterances_audit:])
        
        # Combine split data across all subjects (of interest)
        audit_splits = audit_people.select(np.concatenate(audit_splits))
        train_splits = audit_people.select(np.concatenate(train_splits))
        
        # Do a similar thing for adversary
        # We want to set aside some recordings some person for 'audit' i.e.
        # Running our attacks locally and computing thresholds
        adv_audit_splits, adv_train_splits = [], []
        adv_speaker_ids = adv_data['speaker_id']
        adv_speakers = np.unique(adv_speaker_ids)
        for speaker_id in tqdm(adv_speakers, "Setting aside recordings (adv) for holdout"):
            person_instances = np.where(adv_speaker_ids == speaker_id)[0]
            # Split this into train and audit
            np.random.shuffle(person_instances)
            adv_audit_splits.append(person_instances[:self.hold_per_person_adv])
            adv_train_splits.append(person_instances[self.hold_per_person_adv:])

        # Combine split data across all subjects (adv)
        adv_audit_splits = adv_data.select(np.concatenate(adv_audit_splits))
        adv_train_splits = adv_data.select(np.concatenate(adv_train_splits))

        # Make sure directories exist (for split information)
        os.makedirs(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv"), exist_ok=True)
        os.makedirs(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "victim"), exist_ok=True)

        # Save files
        # For audit people
        audit_splits.save_to_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv", "audit_subjects"))
        train_splits.save_to_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "victim", "audit_subjects"))
        # And adversary's data splits
        adv_audit_splits.save_to_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv", "holdout_subjects"))
        adv_train_splits.save_to_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv", "train_subjects"))
        # If using train.500, also need to save victim data explicitly (since it is a subset of train.500, not train.360 like the case of 'clean')

        # Save info on people to sample for (for victim training)
        np.savetxt(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "victim", "train.txt"), remaining_speaker_ids_indices, delimiter=',')
        np.savetxt(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "audit_speaker_ids.txt"), audit_speaker_ids, delimiter=',')
    
    def prepare_processed_data(self):
        """
            One-time processing to save time when training multiple models
        """
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny.en")

        def process(ds, sampling_rate: int = 16000):
            ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
            def mel(batch):
                audio = batch["audio"]
                batch["input_features"] = feature_extractor(audio["array"], sampling_rate=sampling_rate).input_features[0]
                return batch
            ds = ds.map(mel, num_proc=1)
            return ds

        # Make sure directories exist (for split information)
        os.makedirs(os.path.join(self.base_data_dir, "processed", "victim"), exist_ok=True)
        os.makedirs(os.path.join(self.base_data_dir, "processed", "adv"), exist_ok=True)

        # Adversary's holdout data
        adv_holdout = load_from_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv", "holdout_subjects"))
        adv_holdout = process(adv_holdout)
        adv_holdout.save_to_disk(os.path.join(self.base_data_dir, "processed", "adv", "holdout_subjects"))
        adv_holdout.cleanup_cache_files()
        print("Processed adversary (holdout) data!")
        # Adversary's train data
        adv_train = load_from_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv", "train_subjects"))
        adv_train = process(adv_train)
        adv_train.save_to_disk(os.path.join(self.base_data_dir, "processed", "adv", "train_subjects"))
        adv_train.cleanup_cache_files()
        print("Processed adversary train (use) data!")
        # Adversary's test data
        adv_test = load_dataset("librispeech_asr", self.data_quality_split, split="validation", cache_dir=self.base_data_dir)
        adv_test = process(adv_test)
        adv_test.save_to_disk(os.path.join(self.base_data_dir, "processed", "adv", "test"))
        adv_test.cleanup_cache_files()
        print("Processed adversary test data!")

        # Victim's train data
        if self.data_quality_split == "clean":
            # train.360 is directly victim's train data, not much to think about
            victim_train = load_dataset("librispeech_asr", self.data_quality_split, split="train.360", cache_dir=self.base_data_dir)
        else:
            # victim data is a subset of train.500, so need to load train.500 and filter
            victim_train = load_dataset("librispeech_asr", self.data_quality_split, split="train.500", cache_dir=self.base_data_dir)
            select_indices = np.loadtxt(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "victim", "train.txt"), delimiter=',')
            victim_train = victim_train.select(select_indices)
        victim_train = process(victim_train)
        victim_train.save_to_disk(os.path.join(self.base_data_dir, "processed", "victim", "train"))
        victim_train.cleanup_cache_files()
        print("Processed victim train data!")
        # Victim's test data
        victim_test = load_dataset("librispeech_asr", self.data_quality_split, split="test", cache_dir=self.base_data_dir)
        victim_test = process(victim_test)
        victim_test.save_to_disk(os.path.join(self.base_data_dir, "processed", "victim", "test"))
        victim_test.cleanup_cache_files()
        print("Processed victim test data!")

        # Audit's train data
        audit_train = load_from_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "victim", "audit_subjects"))
        audit_train = process(audit_train)
        audit_train.save_to_disk(os.path.join(self.base_data_dir, "processed", "victim", "audit_subjects"))
        audit_train.cleanup_cache_files()
        print("Processed audit (train) data!")
        # Audit's auditing test
        audit_holdout = load_from_disk(os.path.join(self.base_data_dir, f"splits_person_{self.data_quality_split}", "adv", "audit_subjects"))
        audit_holdout = process(audit_holdout)
        audit_holdout.save_to_disk(os.path.join(self.base_data_dir, "processed", "adv", "audit_subjects"))
        audit_holdout.cleanup_cache_files()
        print("Processed audit (test) data!")


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
    
    def clear_cache(self):
        self.dataset.cleanup_cache_files()

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
                 shuffle_defense: ShuffleDefense = None,
                 matchdg_config: MatchDGConfig = None):
        super().__init__(data_config,
                         skip_data=skip_data,
                         label_noise=label_noise,
                         shuffle_defense=shuffle_defense)
        self.info_object = DatasetInformation(epoch_wise=epoch)

        if self.split == "adv" and self.ratio == 1:
            print(warning_string("\nThis setting only valid for launching attack- make sure not training model with it!\n"))
        if int(self.prop) < 0 or int(self.prop) >= self.info_object.holdout_people:
            raise ValueError(f"Invalid prop: {int(self.prop)}. Must be in [0, {self.info_object.holdout_people})")

        if self.info_object.data_quality_split == "clean":
            self._prop_wise_subsample_sizes = {
                "adv": (200, 30), # Out of (251, 40) [clean]
                "victim": (750, 30) # Out of (901, 40) [clean]
            }
        else:
            self._prop_wise_subsample_sizes = {
                "adv": (200, 25), # Out of (292, 33) [other]
                "victim": (720, 25) # Out of (868, 33) [other]
            }
        self.n_people, self.n_people_test = self._prop_wise_subsample_sizes[self.split]

    """
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
    """

    def load_data(self, model_arch: str):
        base_data_dir = self.info_object.base_data_dir

        train_source = load_from_disk(os.path.join(base_data_dir, "processed", self.split, "train_subjects" if self.split == "adv" else "train"))
        test_source = load_from_disk(os.path.join(base_data_dir, "processed", self.split, "test"))
 
        #  Only need to filter further if 'clean', since 'other' was already filtered earlier
        if self.split == "victim" and self.info_object.data_quality_split == "clean":
            # Pick all non-audit people
            select_indices = np.loadtxt(os.path.join(base_data_dir, f"splits_person_{self.info_object.data_quality_split}", "victim", "train.txt"), delimiter=',')
            train_source = train_source.select(select_indices)
        
        n_people_train = self.n_people
        if self.split == "victim" and self.ratio == 1:
            n_people_train -= 1  # All but one to be sampled from pool of people

        train_speaker_ids = train_source["speaker_id"]
        test_speaker_ids  = test_source["speaker_id"]

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
            audit_speaker_ids = np.loadtxt(os.path.join(base_data_dir, f"splits_person_{self.info_object.data_quality_split}", "audit_speaker_ids.txt"), delimiter=',')
            # Pick the one requested
            wanted_speaker = audit_speaker_ids[int(self.prop)]
            extra_ds_load = load_from_disk(os.path.join(base_data_dir, "processed", self.split, "audit_subjects"))
            # Filter data to get only that speaker
            speaker_ids = extra_ds_load["speaker_id"]
            extra_ds_load = extra_ds_load.select(np.where(speaker_ids == wanted_speaker)[0])

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
        if self.info_object.data_quality_split == "clean":
            # Directly use train.100 (adv) or train.360 (victim)
            train_source = load_dataset(
                "librispeech_asr", self.info_object.data_quality_split, split="train.100" if self.split == "adv" else "train.360",
                cache_dir=self.info_object.base_data_dir)
        else:
            # Load train.500 and short-list based on split information
            train_source = load_dataset(
                "librispeech_asr", self.info_object.data_quality_split, split="train.500",
                cache_dir=self.info_object.base_data_dir)
            select_indices = np.loadtxt(os.path.join(
                self.info_object.base_data_dir, f"splits_person_{self.info_object.data_quality_split}", self.split, "train.txt"), delimiter=',')
            train_source = train_source.select(select_indices)
            
        people_all_train = set(train_source["speaker_id"])
        non_members = people_all_train.difference(set(used_ids))
        return np.array(list(non_members))

    def get_used_indices(self):
        return self.people_in_train, self.people_in_test

    def get_loaders(self, batch_size: int,
                    model_arch: str,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: int = 1,
                    num_workers: int = 2,
                    prefetch_factor: int = 2,
                    pin_memory: bool = True,
                    indexed_data=None):
        self.ds_train, self.ds_val = self.load_data(model_arch)

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
