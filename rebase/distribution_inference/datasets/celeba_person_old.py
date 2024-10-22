import os
from distribution_inference.defenses.active.shuffle import ShuffleDefense
import pandas as pd
from torchvision import transforms
import gc
from PIL import Image
import torch as ch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from typing import List

import distribution_inference.datasets.base as base
import distribution_inference.datasets.utils as utils
import distribution_inference.models.contrastive as models_contrastive
from distribution_inference.config import TrainConfig, DatasetConfig
from distribution_inference.training.utils import load_model
from distribution_inference.utils import model_compile_supported


class DatasetInformation(base.DatasetInformation):
    def __init__(self, epoch_wise: bool = False):
        ratios = [0, 1]
        # 0 (False) here means specified members not present in training data
        # 1 (True) here means specified members present in training data
        super().__init__(name="Celeb-A Person",
                         data_path="celeba",
                         models_path="models_celeba_person/80_20_split",
                         properties=["Subject_MI"],
                         values={"Subject_MI": ratios},
                         supported_models=[
                             "arcface_sungetal", "arcface_resnet"],
                         default_model="arcface_sungetal",
                         epoch_wise=epoch_wise)
        self.holdout_people = 100 # Number of people in holdout set (always part of victim training)
        self.audit_per_person = 10 # Number of images per person in audit set
        self.min_per_person = 20 # Minimum number of images per person in training set
        self.n_gallery_test = 5 # Number of images in gallery for each person in test set

    def get_model(self, parallel: bool = False, fake_relu: bool = False,
                  latent_focus=None, cpu: bool = False,
                  model_arch: str = None,
                  for_training: bool = False,
                  n_people: int = None) -> nn.Module:
        if model_arch is None or model_arch == "None":
            model_arch = self.default_model

        if model_arch == "arcface_sungetal":
            model = models_contrastive.ArcFaceSungetal(n_people=n_people)
        elif model_arch == "arcface_resnet":
            model = models_contrastive.ArcFaceResnet(n_people=n_people)
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
        people = list(set(identities))
        people_shuffled = np.random.permutation(people)
        split_adv = int(len(people_shuffled) * adv_ratio)
        people_adv = people[:split_adv]
        people_victim = people[split_adv:]

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
        # Idea 1: Always to to infer a random self.holdout_people people from training
        # Idea 2: Always include the same self.holdout_people random people in all victim model training,
        # and infer their presence (more streamlined approach)
        # Adversary uses its 20 split to perform shadow training and also use identities 'not part of training'
        # Load metadata files
        splits = self._get_splits()
        ids = self._get_identities()

        # 0 train, 1 validation, 2 test
        train_mask = np.logical_or(
            splits[1].values == 0, splits[1].values == 1)
        test_mask = splits[1].values == 2

        train_map = {p: np.sum(ids[train_mask] == p) for p in np.unique(ids[train_mask])}
        test_map = {p: np.sum(ids[test_mask] == p) for p in np.unique(ids[test_mask])}

        # Discard people from train_map where < self.min_per_person images per person present
        train_map_keep = [k for k, v in train_map.items() if v >= self.min_per_person]
        train_mask = np.logical_and(train_mask, np.isin(ids, train_map_keep))

        # Discard people from test_map where < self.n_gallery_test images per person present
        test_map_keep = [k for k, v in test_map.items() if v > self.n_gallery_test]
        test_mask = np.logical_and(test_mask, np.isin(ids, test_map_keep))

        # Splits on test data
        test_adv_people, test_victim_people = self._victim_adv_identity_split(
            ids[test_mask], adv_ratio=adv_ratio)

        # Splits on train data
        train_adv_people, train_victim_people = self._victim_adv_identity_split(
            ids[train_mask], adv_ratio=adv_ratio)
    
        # Pick random self.holdout_people from train_victim_people
        # These are the people that will be used as targets for ingerring membership in victim model training
        # For 'always used' pick out of people that have at least self.audit_per_person + self.min_per_person images
        # So that after reserving self.audit_per_person images for auditing, we have at least self.min_per_person images for training
        train_victim_always_used = []
        for p in train_victim_people:
            if train_map[p] >= self.audit_per_person + self.min_per_person:
                train_victim_always_used.append(p)
        train_victim_always_used = np.array(train_victim_always_used)
        np.random.shuffle(train_victim_always_used)
        train_victim_always_used = train_victim_always_used[:self.holdout_people]
        train_victim_people = list(set(train_victim_people) - set(train_victim_always_used))

        # For train_victim_always_used, save 'self.holdout_people' images for each person as those not used in training
        # And are esentially available with adversary for 'auditing'
        filenames = np.array(splits.index.tolist())

        # Pick filenames where identity is in wanted_people
        always_used_train, always_used_audit = [], []
        for person in train_victim_always_used:
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
        os.makedirs(os.path.join(self.base_data_dir, "splits_person", "80_20", "adv"), exist_ok=True)
        os.makedirs(os.path.join(self.base_data_dir, "splits_person", "80_20", "victim"), exist_ok=True)

        # Save audit-related information
        save(always_used_train, os.path.join(
            "splits_person", "80_20", "victim", "always_used_train.txt"))
        save(always_used_audit, os.path.join(
            "splits_person", "80_20", "adv", "always_used_audit.txt"))

        # Save generated splits
        save(test_adv_people, os.path.join(
            "splits_person", "80_20", "adv", "test.txt"))
        save(test_victim_people, os.path.join(
            "splits_person", "80_20", "victim", "test.txt"))
        save(train_adv_people, os.path.join(
            "splits_person", "80_20", "adv", "train.txt"))
        save(train_victim_people, os.path.join(
            "splits_person", "80_20", "victim", "train.txt"))


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
                 remap_classes: bool,
                 shuffle: bool = False,
                 transform = None,
                 person_of_interest_indicator: np.ndarray = None):
        super().__init__()
        self.transform = transform
        self.info_object = DatasetInformation()
        self.filenames_list = filenames_list
        self.person_of_interest_indicator = person_of_interest_indicator

        if remap_classes:
            # Transform random labels to (0, n)
            # Keep track of mapping
            self.mapping = make_mapping(labels_list)
            self.labels_list = np.array([self.mapping[p] for p in labels_list])
        else:
            self.labels_list = np.array(labels_list)

        if shuffle:
            shuffle_order = np.arange(len(self.filenames_list))
            np.random.shuffle(shuffle_order)
            self.filenames_list = self.filenames_list[shuffle_order]
            self.labels_list = self.labels_list[shuffle_order]
            if self.person_of_interest_indicator is not None:
                assert len(self.person_of_interest_indicator) == len(self.filenames_list), "Person of interest indicator must be same length as filenames list!"
                self.person_of_interest_indicator = self.person_of_interest_indicator[shuffle_order]

        self.num_samples = len(self.filenames_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # TODO: Third argument keeps track of 'member or not'

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
                 shuffle_defense: ShuffleDefense = None):
        super().__init__(data_config,
                         skip_data=skip_data,
                         label_noise=label_noise,
                         shuffle_defense=shuffle_defense,
                         uses_extra_loader_for_gallery=True)
        self.info_object = DatasetInformation(epoch_wise=epoch)

        if self.split == "adv" and self.ratio == 1:
            raise ValueError("Adversary does not train on data from subject to be inferred (yet- will change later)")
        if int(self.prop) < 0 or int(self.prop) >= self.info_object.holdout_people:
            raise ValueError(f"Invalid prop: {int(self.prop)}. Must be in [0, {self.info_object.holdout_people})")

        # Make sure specified label is valid
        # if self.classify not in self.info_object.preserve_properties:
        #     raise ValueError("Specified label not available for images")

        resize_to = 128 #96
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
        self._prop_wise_subsample_sizes = {
            "adv": (800, 30),
            "victim": (4000, 500)
        }
        self.n_people, self.n_people_test = self._prop_wise_subsample_sizes[self.split]

    def _pick_wanted_people(self, wanted_people_path: str, n_people: int):
        with open(wanted_people_path, 'r') as f:
            wanted_people = f.read().splitlines()
        wanted_people = [int(x) for x in wanted_people]
        # Sub-sample desired number of people
        wanted_people = np.random.choice(
            wanted_people, n_people, replace=False)

        identities = self.info_object._get_identities()
        splits = self.info_object._get_splits()
        filenames = np.array(splits.index.tolist())

        # Pick filenames where identity is in wanted_people
        mask = np.isin(identities, wanted_people)
        filenames = filenames[mask]
        identities = identities[mask]
        
        # Essentially (path, label)
        return filenames, identities

    def _gallery_split(self, filenames, labels):
        """
            Takes test-faces and splits them into gallery and non-gallery
            images. Gallery images are used to compute embeddings, which are then compared
            with a given test image to make class prediction.
        """
        unique_labels = np.unique(labels)
        filenames_gallery, labels_gallery, filenames_non_gallery, labels_non_gallery = [], [], [], []
        for u in unique_labels:
            matching_files = filenames[labels == u]
            # Sort (do not want randomness here)
            matching_files = np.sort(matching_files)

            if len(matching_files) <= self.info_object.n_gallery_test:
                raise ValueError(f"Not enough images for gallery: {self.info_object.n_gallery_test} requested, {len(matching_files)} available")

            filenames_gallery.append(matching_files[:self.info_object.n_gallery_test])
            labels_gallery.append(np.array([u] * self.info_object.n_gallery_test))
            filenames_non_gallery.append(matching_files[self.info_object.n_gallery_test:])
            labels_non_gallery.append(np.array([u] * (len(matching_files) - self.info_object.n_gallery_test)))

        filenames_gallery = np.concatenate(filenames_gallery)
        labels_gallery = np.concatenate(labels_gallery)
        filenames_non_gallery = np.concatenate(filenames_non_gallery)
        labels_non_gallery = np.concatenate(labels_non_gallery)

        mapping = {p: i for i, p in enumerate(unique_labels)}
        labels_gallery = np.array([mapping[p] for p in labels_gallery])
        labels_non_gallery = np.array([mapping[p] for p in labels_non_gallery])

        return (filenames_gallery, labels_gallery), (filenames_non_gallery, labels_non_gallery)

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

    def load_data(self):
        # Use relevant file split information
        people_list_train = os.path.join(
            self.info_object.base_data_dir,
            "splits_person", "80_20", self.split, "train.txt")
        people_list_test = os.path.join(
            self.info_object.base_data_dir,
            "splits_person", "80_20", self.split, "test.txt")
        
        n_people_train = self.n_people
        if self.split == "victim" and self.ratio == 1:
            n_people_train -= 1 # All but one to be sampled from pool of people    
        # if self.split == "adv":
        #     people_always_used = os.path.join(
        #         self.info_object.base_data_dir,
        #         "splits_person", "80_20", "adv", "always_used_audit.txt")

        # Adjust number of people to sample from pool
        filenames_train, labels_train = self._pick_wanted_people(
            people_list_train, n_people_train)
        filenames_test, labels_test = self._pick_wanted_people(
            people_list_test, self.n_people_test)

        person_of_interest_indicator = None
        if self.split == "victim" and self.ratio == 1:
            people_always_used = os.path.join(
                self.info_object.base_data_dir,
                "splits_person", "80_20", "victim", "always_used_train.txt")
            filenames_always_train, labels_always_train = self._load_data_for_always_included_people(people_always_used)
            if len(set(labels_train).intersection(set(labels_always_train))) != 0:
                raise ValueError("Intersection between train and always_used_train is not empty- this should not happen!")
            filenames_train = np.concatenate((filenames_train, filenames_always_train))
            person_of_interest_indicator = np.zeros(len(filenames_train))
            person_of_interest_indicator[-len(filenames_always_train):] = 1
            labels_train = np.concatenate((labels_train, labels_always_train))

        # For test, define gallery-non-gallery split
        (filenames_test_gallery, labels_test_gallery), (filenames_test_use, labels_test_use) = self._gallery_split(filenames_test, labels_test)

        # Create datasets (let DS objects handle mapping of labels)
        ds_train = CelebAPerson(
            filenames_train, labels_train,
            remap_classes=True,
            transform=self.train_transforms,
            person_of_interest_indicator=person_of_interest_indicator)

        # _gallery_split ensures that the two sets (gallery and test) have consistent labels
        ds_test = CelebAPerson(
            filenames_test_use, labels_test_use,
            remap_classes=False,
            transform=self.test_transforms)
        ds_test_gallery = CelebAPerson(
            filenames_test_gallery, labels_test_gallery,
            remap_classes=False,
            transform=self.test_transforms)

        return ds_train, ds_test, ds_test_gallery

    def get_loaders(self, batch_size: int,
                    shuffle: bool = True,
                    eval_shuffle: bool = False,
                    val_factor: int = 2,
                    num_workers: int = 8,
                    prefetch_factor: int = 2,
                    indexed_data=None):
        self.ds_train, self.ds_val, self.ds_val_gallery = self.load_data()

        return super().get_loaders(batch_size, shuffle=shuffle,
                                   eval_shuffle=eval_shuffle,
                                   val_factor=val_factor,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch_factor)

    def get_save_dir(self, train_config: TrainConfig, model_arch: str) -> str:
        base_models_dir = self.info_object.base_models_dir
        subfolder_prefix = os.path.join(
            self.split, self.prop, str(self.ratio)
        )
        if not (train_config.misc_config and train_config.misc_config.contrastive_config):
            raise ValueError("Only contrastive training is supported for this dataset")
        else:
            contrastive_config = train_config.misc_config.contrastive_config

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
