import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import pandas as pd
from torchvision import transforms
from PIL import Image
import torch as ch
import numpy as np
import torch.nn as nn
from typing import List

import distribution_inference.datasets.base as base
import distribution_inference.models.contrastive as models_contrastive
from distribution_inference.config import TrainConfig, DatasetConfig, MatchDGConfig
from distribution_inference.training.utils import load_model
from distribution_inference.utils import model_compile_supported
from distribution_inference.datasets._contrastive_utils import NWays, KShots, LoadData, RemapLabels, TaskDataset, MetaDataset
from distribution_inference.utils import warning_string


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0, 1]
        # 0 (False) here means specified members not present in training data
        # 1 (True) here means specified members present in training data
        holdout_people = 100
        props = [str(x) for x in range(holdout_people)]
        super().__init__(name="Celeb-A Person",
                         data_path="celeba",
                         models_path="models_celeba_person/50_50_split",
                         properties=props,
                         values={k: ratios for k in props},
                         supported_models=["scnn_relation", "scnn_deeper_relation"],
                         default_model="scnn_relation",
                         epoch_wise=epoch_wise,
                         user_level_mi=True)
        self.holdout_people = holdout_people # Number of people in holdout set (always part of victim training)
        self.audit_per_person = 10 # Number of images per person in audit set
        self.min_per_person = 20 # Minimum number of images per person in training set
        self.hold_per_person_adv = 5 # Number of image per person (in adv) not to use for training
        # Will be useful when computing metrics for audits, since we don't want to
        # use exact records for in/out subject comparisons

    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, cpu: bool = False,
                  model_arch: str = None,
                  for_training: bool = False) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model

        if model_arch == "scnn_relation":
            model = models_contrastive.SCNNFaceAudit()
        elif model_arch == "scnn_deeper_relation":
            model = models_contrastive.SCNNDeeperFaceAudit()
        else:
            raise NotImplementedError("Model architecture not supported")

        if parallel:
            model = nn.DataParallel(model)
        if not cpu:
            model = model.cuda()
        
        if for_training and model_compile_supported():
            model = ch.compile(model)

        return model

    def _victim_adv_identity_split(self, identities, adv_ratio: float):
        # Extract unique people
        people = np.unique(identities)
        # Shuffle them
        people_shuffled = np.random.permutation(people)
        # Get 'people used by adv' from this
        split_adv = int(len(people_shuffled) * adv_ratio)
        people_adv = people_shuffled[:split_adv]
        # And the rest to be used by victim
        people_victim = people_shuffled[split_adv:]

        return people_adv, people_victim

    def _get_splits(self):
        fpath = os.path.join(self.base_data_dir, "list_eval_partition.txt")
        splits = pd.read_csv(fpath, delim_whitespace=True,
                                header=None, index_col=0)
        return splits

    def _get_identities(self):
        fpath = os.path.join(self.base_data_dir, "identity_CelebA.txt")
        identity = pd.read_csv(fpath, delim_whitespace=True,
                               header=None, index_col=0)
        return np.array(identity.values).squeeze(1)

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = 0.2,
                                         test_ratio=None):
        """
            Generate and store data offline for victim and adversary
            using the given dataset. Use this method only once for the
            same set of experiments.
        """

        # 10,177 identities in total
        # Make a 80:20 split of victim:adversary identities
        # Adversary uses its 20 split to perform shadow training and/or launching attacks
        # Load metadata files
        splits = self._get_splits()
        ids = self._get_identities()

        # 0 train, 1 validation, 2 test
        train_mask = np.logical_or(
            splits[1].values == 0, splits[1].values == 1)
        test_mask = splits[1].values == 2

        vals, counts = np.unique(ids[train_mask], return_counts=True)
        train_map = dict(zip(vals, counts))
        vals, counts = np.unique(ids[test_mask], return_counts=True)
        test_map = dict(zip(vals, counts))

        # Discard people where < self.min_per_person images per person present
        train_map_keep = [k for k, v in train_map.items() if v >= self.min_per_person]
        train_mask = np.isin(ids, train_map_keep)
        test_map_keep = [k for k, v in test_map.items() if v >= self.min_per_person]
        test_mask = np.isin(ids, test_map_keep)

        # Splits on test data
        test_adv_people, test_victim_people = self._victim_adv_identity_split(
            ids[test_mask], adv_ratio=adv_ratio)

        # Splits on train data
        train_adv_people, train_victim_people = self._victim_adv_identity_split(
            ids[train_mask], adv_ratio=adv_ratio)
    
        # Pick random self.holdout_people from train_victim_people
        # These are the people that will be used as targets for inferring membership in victim model training
        # For 'always used' pick out of people that have at least self.audit_per_person + self.min_per_person images
        # So that after reserving self.audit_per_person images for auditing, we have at least self.min_per_person images for training
        holdout_people_data = []
        for p in train_victim_people:
            if train_map[p] >= self.audit_per_person + self.min_per_person:
                holdout_people_data.append(p)
        holdout_people_data = np.array(holdout_people_data)
        np.random.shuffle(holdout_people_data)
        holdout_people_data = holdout_people_data[:self.holdout_people]
        train_victim_people = list(set(train_victim_people) - set(holdout_people_data))

        # For holdout_people_data, save 'self.holdout_people' images for each person as those not used in training
        # And are esentially available with adversary for 'auditing'
        filenames = np.array(splits.index.tolist())

        # Pick filenames where identity is in wanted_people
        always_used_train, always_used_audit = [], []
        for person in holdout_people_data:
            shortlisted = filenames[ids == person]
            np.random.shuffle(shortlisted)
            for_training = shortlisted[self.audit_per_person:]
            # Save images to be used for training
            for f in for_training:
                always_used_train.append("%s,%d" % (f, person))
            # Save images to be used for auditing
            for_auditing = shortlisted[:self.audit_per_person]
            for f in for_auditing:
                always_used_audit.append("%s,%d" % (f, person))

        # Save split files for later use
        def save(data, path):
            with open(os.path.join(self.base_data_dir, path), 'w') as f:
                f.writelines("%s\n" % l for l in data)

        # Make sure directories exist (for split information)
        os.makedirs(os.path.join(self.base_data_dir, "splits_person", "50_50", "adv"), exist_ok=True)
        os.makedirs(os.path.join(self.base_data_dir, "splits_person", "50_50", "victim"), exist_ok=True)

        # Save audit-related information
        save(always_used_train, os.path.join(
            "splits_person", "50_50", "victim", "always_used_train.txt"))
        save(always_used_audit, os.path.join(
            "splits_person", "50_50", "adv", "always_used_audit.txt"))

        # Save generated splits
        save(test_adv_people, os.path.join(
            "splits_person", "50_50", "adv", "test.txt"))
        save(test_victim_people, os.path.join(
            "splits_person", "50_50", "victim", "test.txt"))
        save(train_adv_people, os.path.join(
            "splits_person", "50_50", "adv", "train.txt"))
        save(train_victim_people, os.path.join(
            "splits_person", "50_50", "victim", "train.txt"))


def make_mapping(labels):
    """
        Map an arbitrary list of people to [0, n)
    """
    unique_people = np.unique(labels)
    mapping = {p: i for i, p in enumerate(unique_people)}
    return mapping


class CelebAPerson(base.CustomDataset):
    def __init__(self, filenames_list,
                 labels_list,
                 shuffle: bool = False,
                 transform = None,
                 person_of_interest_indicator: np.ndarray = None):
        super().__init__()
        self.transform = transform
        self.info_object = DatasetInformation()
        self.filenames_list = filenames_list
        self.person_of_interest_indicator = person_of_interest_indicator

        if self.person_of_interest_indicator is not None:
            assert len(self.person_of_interest_indicator) == len(self.filenames_list), "Person of interest indicator must be same length as filenames list!"

        self.labels_list = np.array(labels_list)

        if shuffle:
            shuffle_order = np.arange(len(self.filenames_list))
            np.random.shuffle(shuffle_order)
            self.filenames_list = self.filenames_list[shuffle_order]
            self.labels_list = self.labels_list[shuffle_order]
            if self.person_of_interest_indicator is not None:
                self.person_of_interest_indicator = self.person_of_interest_indicator[shuffle_order]

        self.num_samples = len(self.filenames_list)

    def __len__(self):
        return self.num_samples

    def get_labels_list(self):
        return self.labels_list

    def __getitem__(self, idx):
        # Open image
        filename = self.filenames_list[idx]
        x = Image.open(os.path.join(
            self.info_object.base_data_dir, "img_align_celeba", filename))
        if self.transform:
            x = self.transform(x)

        y = self.labels_list[idx] 
        indicator = self.person_of_interest_indicator[idx] if self.person_of_interest_indicator is not None else 0

        return x, y, indicator


class CelebaPersonWrapper(base.CustomDatasetWrapper):
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

        # Make sure specified label is valid
        # if self.classify not in self.info_object.preserve_properties:
        #     raise ValueError("Specified label not available for images")

        resize_to = 96
        train_transforms = [
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        self.test_transforms = transforms.Compose([
            transforms.Resize((resize_to, resize_to)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        if self.augment:
            augment_transforms = [
                transforms.RandomAffine(degrees=15,
                                        translate=(0.1, 0.1),
                                        shear=0.1),
                transforms.RandomHorizontalFlip()
            ]
            train_transforms = augment_transforms + train_transforms
        self.train_transforms = transforms.Compose(train_transforms)

        # Define number of people to pick for train, test
        # Below is valid for 50:50 victim/adv splits
        self._prop_wise_subsample_sizes = {
            "adv": (2500, 200),
            "victim": (2500, 200)
        }
        # self._prop_wise_subsample_sizes = {
        #     "adv": (800, 50),
        #     "victim": (3500, 300)
        # }
        self.n_people, self.n_people_test = self._prop_wise_subsample_sizes[self.split]

    def _pick_wanted_people(self, wanted_people_path: str, n_people: int):
        with open(wanted_people_path, 'r') as f:
            wanted_people = f.read().splitlines()
        wanted_people = [int(x) for x in wanted_people]
        # Sub-sample desired number of people
        if n_people > len(wanted_people):
            raise ValueError(
                f"Number of people requested ({n_people}) is greater than number of people in file ({len(wanted_people)})")
        wanted_people = np.random.choice(
            wanted_people, n_people, replace=False)
        return self._pick_these_people(wanted_people)

    def _pick_these_people(self, these_people: List[int]):
        """
            Pick specified people and their data
        """
        identities = self.info_object._get_identities()
        splits = self.info_object._get_splits()
        filenames = np.array(splits.index.tolist())

        # Pick filenames where identity is in wanted_people
        mask = np.isin(identities, these_people)
        filenames = filenames[mask]
        identities = identities[mask]

        # Essentially (path, label)
        return filenames, identities

    def _load_data_for_always_included_people(self, always_used_people_path: str):
        paths, labels = [], []
        with open(always_used_people_path, 'r') as f:
            for line in f:
                path, label = line.rstrip('\n').split(',')
                paths.append(path)
                labels.append(int(label))
        sorted_unique_people = np.sort(np.unique(labels))

        if len(sorted_unique_people) != self.info_object.holdout_people:
            raise ValueError(f"Expected {self.info_object.holdout_people} always included people, got {len(sorted_unique_people)}")

        # Pick person according to self.prop
        wanted_person = sorted_unique_people[int(self.prop)]
        paths, labels = np.array(paths), np.array(labels)
        paths = paths[labels == wanted_person]
        labels = labels[labels == wanted_person]
        return paths, labels

    def load_specified_data(self, people_ids: List[int],
                            not_in_train: bool = False,
                            strictly_in_train: bool = False,
                            n_pick: int = None):
        if not_in_train and strictly_in_train:
            raise ValueError("Cannot be both not_in_train and strictly_in_train")
        # Adjust number of people to sample from pool
        filenames, labels = self._pick_these_people(people_ids)

        if not_in_train or strictly_in_train or (n_pick is not None):
            # Only pick images for these people that were NOT used in training the model
            filenames_new, labels_new = [], []
            for identity in np.unique(labels):
                filenames_ = sorted(filenames[labels == identity])
                if not_in_train:
                    filenames_ = filenames_[:self.info_object.hold_per_person_adv]
                elif strictly_in_train:
                    filenames_ = filenames_[self.info_object.hold_per_person_adv:]
                
                if n_pick is not None:
                    # Pick n_pick random images per person
                    # from already-shortlisted images
                    filenames_ = np.random.choice(filenames_, n_pick, replace=False)

                filenames_new.extend(filenames_)
                labels_new.extend([identity] * len(filenames_))
            filenames, labels = np.array(filenames_new), np.array(labels_new)

        ds_use = CelebAPerson(
            filenames, labels,
            shuffle=True,
            transform=self.test_transforms)

        return ds_use

    def load_data(self, primed_for_training: bool = True):
        # Use relevant file split information
        people_list_train = os.path.join(
            self.info_object.base_data_dir,
            "splits_person", "50_50", self.split, "train.txt")
        people_list_test = os.path.join(
            self.info_object.base_data_dir,
            "splits_person", "50_50", self.split, "test.txt")
        
        n_people_train = self.n_people
        if self.split == "victim" and self.ratio == 1:
            n_people_train -= 1 # All but one to be sampled from pool of people    

        # Adjust number of people to sample from pool
        filenames_train, labels_train = self._pick_wanted_people(
            people_list_train, n_people_train)
        filenames_test, labels_test = self._pick_wanted_people(
            people_list_test, self.n_people_test)

        # For each person, get filenames (sorted) and skip the first self.info_object.hold_per_person_adv records per person
        # which will be used in audit mode (attack) but not actual training. Sorting ensures deterministic sampling of 
        # records when training or auditing
        if self.split == "adv":
            filenames_train_new, labels_train_new = [], []
            for identity in np.unique(labels_train):
                filenames = sorted(filenames_train[labels_train == identity])
                # If for training:
                filenames = filenames[self.info_object.hold_per_person_adv:]
                filenames_train_new.extend(filenames)
                labels_train_new.extend([identity] * len(filenames))
            filenames_train, labels_train = np.array(filenames_train_new), np.array(labels_train_new)

        person_of_interest_indicator = None
        if self.split == "victim" and self.ratio == 1:
            # Will be part of training
            people_always_used = os.path.join(
                self.info_object.base_data_dir,
                "splits_person", "50_50", "victim", "always_used_train.txt")
            filenames_always_train, labels_always_train = self._load_data_for_always_included_people(people_always_used)
            if len(set(labels_train).intersection(set(labels_always_train))) != 0:
                raise ValueError("Intersection between train and always_used_train is not empty- this should not happen!")
            filenames_train = np.concatenate((filenames_train, filenames_always_train))
            person_of_interest_indicator = np.zeros(len(filenames_train))
            person_of_interest_indicator[-len(filenames_always_train):] = 1
            labels_train = np.concatenate((labels_train, labels_always_train))
        elif self.split == "adv" and self.ratio == 1:
            # Will be part of testing (for auditing)
            people_always_used = os.path.join(
                self.info_object.base_data_dir,
                "splits_person", "50_50", "adv", "always_used_audit.txt")
            filenames_always_test, labels_always_test = self._load_data_for_always_included_people(people_always_used)
            filenames_test = np.concatenate((filenames_test, filenames_always_test))
            person_of_interest_indicator = np.zeros(len(filenames_test))
            person_of_interest_indicator[-len(filenames_always_test):] = 1
            labels_test = np.concatenate((labels_test, labels_always_test))

        # Keep note of people (identifiers) used in training and validation for this specific instance
        self.people_in_train = np.unique(labels_train)
        self.people_in_test = np.unique(labels_test)

        # Create datasets (let DS objects handle mapping of labels)
        ds_train = CelebAPerson(
            filenames_train, labels_train,
            shuffle=True,
            transform=self.train_transforms,
            person_of_interest_indicator=person_of_interest_indicator if self.split == "victim" else None)

        ds_test = CelebAPerson(
            filenames_test, labels_test,
            shuffle=True,
            transform=self.test_transforms,
            person_of_interest_indicator=person_of_interest_indicator if self.split == "adv" else None)
        
        if not primed_for_training:
            return ds_train, ds_test

        ds_train = MetaDataset(ds_train)
        ds_test  = MetaDataset(ds_test)

        self.train_transforms_task = [
            NWays(ds_train, self.relation_config.n_way),
            KShots(ds_train, self.relation_config.num_query_train + self.relation_config.k_shot),
            LoadData(ds_train),
            RemapLabels(ds_train)
        ]
        self.test_transforms_task = [
            NWays(ds_test, self.relation_config.n_way),
            KShots(ds_test, self.relation_config.num_query_test + self.relation_config.k_shot),
            LoadData(ds_test),
            RemapLabels(ds_test)
        ]

        train_dset = TaskDataset(ds_train,
                                task_transforms=self.train_transforms_task)
        test_dset  = TaskDataset(ds_test,
                                task_transforms=self.test_transforms_task,
                                num_tasks=self.relation_config.test_num_task)

        return train_dset, test_dset

    def get_non_members(self, used_ids: List[int]):
        # Use relevant file split information
        people_list_train = os.path.join(
            self.info_object.base_data_dir,
            "splits_person", "50_50", self.split, "train.txt")
        with open(people_list_train, 'r') as f:
            wanted_people = f.read().splitlines()
        wanted_people = set([int(x) for x in wanted_people])
        non_members = wanted_people.difference(set(used_ids))
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
                    primed_for_training: bool=True,
                    indexed_data=None):
        self.ds_train, self.ds_val = self.load_data(primed_for_training=primed_for_training)

        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   val_factor=val_factor,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   prefetch_factor=prefetch_factor)

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        subfolder_prefix = os.path.join(
            self.split, self.prop, str(self.ratio)
        )
        if not (train_config.data_config.relation_config):
            raise ValueError("Only relation-net training is supported for this dataset")

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
