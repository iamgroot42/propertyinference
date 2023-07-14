"""
    Primarily used as a dataset for transfer setting in CASIA-WebFace.
    Attacker trains its models on this dataset (varying dataset sizes), where the
    property (size) indicates the ratio of dataset used
"""
import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
from torchvision import transforms
import pickle
from PIL import Image
import torch as ch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from typing import List

import distribution_inference.datasets.base as base
import distribution_inference.models.contrastive as models_contrastive
from distribution_inference.config import TrainConfig, DatasetConfig, MatchDGConfig
from distribution_inference.training.utils import load_model
from distribution_inference.utils import model_compile_supported
from distribution_inference.datasets._contrastive_utils import NWays, KShots, LoadData, RemapLabels, TaskDataset, MetaDataset


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        super().__init__(name="MAADFace-Person",
                         data_path="vggface2",
                         models_path="models_maadface_person",
                         properties=['size'],
                         values={'size': [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]},
                         supported_models=["scnn_relation"],
                         default_model="scnn_relation",
                         epoch_wise=epoch_wise)
        self.audit_per_person = 10  # Number of images per person in audit set
        # Not needed for this dataset: all users have >= 87 images
        self.min_per_person = 20  # Minimum number of images per person in training set
        self.holdout_people = 500 # Number of people in holdout (never used in training)
        
    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, cpu: bool = False,
                  model_arch: str = None,
                  for_training: bool = False) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model

        if model_arch == "scnn_relation":
            model = models_contrastive.SCNNFaceAudit()
        else:
            raise NotImplementedError("Model architecture not supported")

        if parallel:
            model = nn.DataParallel(model)
        if not cpu:
            model = model.cuda()

        if for_training and model_compile_supported():
            model = ch.compile(model)

        return model

    def generate_victim_adversary_splits(self,
                                         adv_ratio: float = None,
                                         test_ratio: float = None):
        """
            Generate train-test splits
            Since this is used exclusively for shadow model training, there 
            are no victim/adv splits
        """
        # Open file named 'test_names.txt' and read the names of the test people
        with open(os.path.join(self.base_data_dir, "test_names.txt"), "r") as f:
            test_names = f.readlines()
        # Do same thing for train names
        with open(os.path.join(self.base_data_dir, "train_names.txt"), "r") as f:
            train_names = f.readlines()
        # Remove the newline character from the end of each name
        all_names_train = np.array([name.strip() for name in train_names])
        test_people     = np.array([name.strip() for name in test_names])

        # Shuffle people
        np.random.shuffle(all_names_train)

        # Set aside holdout people
        num_holdout = self.holdout_people
        holdout_people = all_names_train[:num_holdout]
        train_people = all_names_train[num_holdout:]

        # Save split files for later use
        def save(data, path):
            with open(os.path.join(self.base_data_dir, path), 'w') as f:
                f.writelines("%s\n" % l for l in data)

        # Save generated splits
        save(train_people, os.path.join("splits_person", "splits_train.txt"))
        save(test_people, os.path.join("splits_person", "splits_test.txt"))
        save(holdout_people, os.path.join("splits_person", "splits_holdout.txt"))


class MAADPerson(base.CustomDataset):
    def __init__(self, filenames_list,
                 labels_list,
                 shuffle: bool = False,
                 transform=None):
        super().__init__()
        self.transform = transform
        self.info_object = DatasetInformation()
        self.filenames_list = filenames_list
        self.labels_list = np.array(labels_list)

        if shuffle:
            shuffle_order = np.arange(len(self.filenames_list))
            np.random.shuffle(shuffle_order)
            self.filenames_list = self.filenames_list[shuffle_order]
            self.labels_list = self.labels_list[shuffle_order]

        self.num_samples = len(self.filenames_list)

    def __len__(self):
        return self.num_samples

    def get_labels_list(self):
        return self.labels_list

    def __getitem__(self, idx):
        # Open image
        filename = self.filenames_list[idx]
        x = Image.open(os.path.join(
            self.info_object.base_data_dir, "data", filename))
        if self.transform:
            x = self.transform(x)

        y = self.labels_list[idx]
        # Parse to remove first 'n' character and get a numeric label
        y = int(y[1:])

        return x, y, 0


class MAADPersonWrapper(base.CustomDatasetWrapper):
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

        # self.ratio for this dataset corresponds to picking # of users inside dataset
        # self.split is meaningless, so ignore

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

    def _pick_wanted_people(self, wanted_people: List[str], n_people: int = None, subdir: str = "train"):
        # Sub-sample desired number of people
        if n_people is not None and n_people > len(wanted_people):
            raise ValueError(
                f"Number of people requested ({n_people}) is greater than number of people in file ({len(wanted_people)})")
        if n_people is not None:
            wanted_people = np.random.choice(wanted_people, n_people, replace=False)
        return self._pick_these_people(wanted_people, subdir=subdir)

    def _pick_these_people(self, these_people: List[str], subdir: str = "train"):
        """
            Pick specified people and their data
        """
        filenames, identities = [], []

        for identity in these_people:
            within = os.listdir(os.path.join(self.info_object.base_data_dir, subdir, identity))
            # Make paths absolute (relative to data directory)
            within = [os.path.join(self.info_object.base_data_dir, subdir, identity, x) for x in within]
            filenames.extend(within)
            identities.extend([identity] * len(within))
        filenames = np.array(filenames)
        identities = np.array(identities)

        # Essentially (path, label)
        return filenames, identities

    def load_specified_data(self, people_ids: List[int],
                            not_in_train: bool = False,
                            strictly_in_train: bool = False,
                            n_pick: int = None):
        if not_in_train and strictly_in_train:
            raise ValueError(
                "Cannot be both not_in_train and strictly_in_train")
        # Adjust number of people to sample from pool
        filenames, labels = self._pick_these_people(people_ids)

        if not_in_train or strictly_in_train or (n_pick is not None):
            # Only pick images for these people that were NOT used in training the model
            filenames_new, labels_new = [], []
            for identity in np.unique(labels):
                filenames_ = sorted(filenames[labels == identity])
                if not_in_train:
                    filenames_ = filenames_[
                        :self.info_object.audit_per_person]
                elif strictly_in_train:
                    filenames_ = filenames_[
                        self.info_object.audit_per_person:]

                if n_pick is not None:
                    # Pick n_pick random images per person
                    # from already-shortlisted images
                    filenames_ = np.random.choice(
                        filenames_, n_pick, replace=False)

                filenames_new.extend(filenames_)
                labels_new.extend([identity] * len(filenames_))
            filenames, labels = np.array(filenames_new), np.array(labels_new)

        ds_use = MAADPerson(
            filenames, labels,
            shuffle=True,
            transform=self.test_transforms)

        return ds_use

    def load_data(self, primed_for_training: bool = True):
        # Use relevant file split information
        people_list_train = read_names_file(os.path.join(
            self.info_object.base_data_dir,
            "splits_person",
            "splits_train.txt"))
        people_list_test = read_names_file(os.path.join(
            self.info_object.base_data_dir,
            "splits_person",
            "splits_test.txt"))

        # Number of people in train set
        n_people_train = int(self.ratio * len(people_list_train))
        # Number of people in test set (always remains the same)

        # Adjust number of people to sample from pool
        filenames_train, labels_train = self._pick_wanted_people(people_list_train, n_people_train)
        filenames_test, labels_test = self._pick_wanted_people(people_list_test, subdir="test")

        # For each person, get filenames (sorted) and skip the first self.info_object.audit_per_person records per person
        # which will be used in audit mode (attack) but not actual training. Sorting ensures deterministic sampling of
        # records when training or auditing
        filenames_train_new, labels_train_new = [], []
        for identity in np.unique(labels_train):
            filenames = sorted(filenames_train[labels_train == identity])
            # If for training:
            filenames = filenames[self.info_object.audit_per_person:]
            filenames_train_new.extend(filenames)
            labels_train_new.extend([identity] * len(filenames))
        filenames_train, labels_train = np.array(filenames_train_new), np.array(labels_train_new)

        # Keep note of people (identifiers) used in training and validation for this specific instance
        self.people_in_train = np.unique(labels_train)
        self.people_in_test = np.unique(labels_test)

        # Create datasets (let DS objects handle mapping of labels)
        ds_train = MAADPerson(
            filenames_train, labels_train,
            shuffle=True,
            transform=self.train_transforms)

        ds_test = MAADPerson(
            filenames_test, labels_test,
            shuffle=True,
            transform=self.test_transforms)

        if not primed_for_training:
            return ds_train, ds_test

        ds_train = MetaDataset(ds_train)
        ds_test = MetaDataset(ds_test)

        self.train_transforms_task = [
            NWays(ds_train, self.relation_config.n_way),
            KShots(ds_train, self.relation_config.num_query_train +
                   self.relation_config.k_shot),
            LoadData(ds_train),
            RemapLabels(ds_train)
        ]
        self.test_transforms_task = [
            NWays(ds_test, self.relation_config.n_way),
            KShots(ds_test, self.relation_config.num_query_test +
                   self.relation_config.k_shot),
            LoadData(ds_test),
            RemapLabels(ds_test)
        ]

        train_dset = TaskDataset(ds_train,
                                 task_transforms=self.train_transforms_task)
        test_dset = TaskDataset(ds_test,
                                task_transforms=self.test_transforms_task,
                                num_tasks=self.relation_config.test_num_task)

        return train_dset, test_dset

    def get_non_members(self, used_ids: List[int]):
        # Use splits_holdout.txt - none of these members are used in training
        people_list_train = read_names_file(os.path.join(
            self.info_object.base_data_dir,
            "splits_person",
            "splits_holdout.txt"))

        with open(people_list_train, 'r') as f:
            wanted_people = f.read().splitlines()
        non_members = list([x for x in wanted_people])
        return np.array(non_members)

    def get_used_indices(self):
        return self.people_in_train, self.people_in_test

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: int = 1,
                    num_workers: int = 2,
                    prefetch_factor: int = 2,
                    pin_memory: bool = True,
                    primed_for_training: bool = True,
                    indexed_data=None):
        self.ds_train, self.ds_val = self.load_data(
            primed_for_training=primed_for_training)

        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   val_factor=val_factor,
                                   num_workers=num_workers,
                                   pin_memory=pin_memory,
                                   prefetch_factor=prefetch_factor)

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        subfolder_prefix = os.path.join(self.prop, str(self.ratio))
        if not (train_config.data_config.relation_config):
            raise ValueError(
                "Only relation-net training is supported for this dataset")

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


# Helper function to read splits files
def read_names_file(x):
    with open(x, 'r') as f:
        lines = f.read().splitlines()
    return lines
