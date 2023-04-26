"""
    Taken from https://github.com/MinChen00/Face-Auditor/
    Modified to fit with our package. Credits for this file go the the original authors.
"""

from torch.utils.data import Dataset
from collections import defaultdict
import random
import collections
import functools
import copy
import array
from torch.utils.data._utils import collate
import itertools
import numpy as np
from tqdm import tqdm


class MetaDataset(Dataset):
    """
        **Description**

        Wraps a classification dataset to enable fast indexing of samples within classes.

        This class exposes two attributes specific to the wrapped dataset:

        * `labels_to_indices`: maps a class label to a list of sample indices with that label.
        * `indices_to_labels`: maps a sample index to its corresponding class label.

        Those dictionary attributes are often used to quickly create few-shot classification tasks.
        They can be passed as arguments upon instantiation, or automatically built on-the-fly.

        Note that if only one of `labels_to_indices` or `indices_to_labels` is provided, this class builds the other one from it.

        **Arguments**

        * **dataset** (Dataset) -  A torch Dataset.
        * **labels_to_indices** (dict, **optional**, default=None) -  A dictionary mapping labels to the indices of their samples.
        * **indices_to_labels** (dict, **optional**, default=None) -  A dictionary mapping sample indices to their corresponding label.

        **Example**
        ~~~python
        mnist = torchvision.datasets.MNIST(root="/tmp/mnist", train=True)
        mnist = l2l.data.MetaDataset(mnist)
        ~~~
    """

    def __init__(self, dataset: Dataset,
                 labels_to_indices: dict=None,
                 indices_to_labels: dict=None,
                 transform=None):

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "MetaDataset only accepts a torch dataset as input")

        self.dataset = dataset
        self.transform = transform
        
        self.create_bookkeeping(
            labels_to_indices=labels_to_indices,
            indices_to_labels=indices_to_labels)

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self,
                           labels_to_indices: dict=None,
                           indices_to_labels: dict=None):
        """
        Iterates over the entire dataset and creates a map of target to indices.

        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        # Bootstrap from arguments
        if labels_to_indices is not None:
            indices_to_labels = {
                idx: label
                for label, indices in labels_to_indices.items()
                for idx in indices
            }
        elif indices_to_labels is not None:
            labels_to_indices = defaultdict(list)

            for idx, label in indices_to_labels.items():
                labels_to_indices[label].append(idx)
        else:  # Create from scratch
            label_list = self.dataset.get_labels_list()
            indices_to_labels = dict(enumerate(label_list))
            unique_labels = np.unique(label_list)
            labels_to_indices = {label: np.where(label_list == label)[0] for label in unique_labels}            

        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())

        self._bookkeeping = {
            'labels_to_indices': self.labels_to_indices,
            'indices_to_labels': self.indices_to_labels,
            'labels': self.labels
        }


class DataDescription:
    def __init__(self, index):
        self.index = index
        self.transforms = []


class TaskTransform:
    def __init__(self, dataset: MetaDataset):
        if type(dataset) != MetaDataset:
            raise ValueError("TaskTransforms only work on MetaDatasets")
        self.dataset = dataset

    def new_task(self):
        n = len(self.dataset)
        task_description = [None] * n
        for i in range(n):
            task_description[i] = DataDescription(i)
        return task_description


class NWays(TaskTransform):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps samples from N random labels present in the task description.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.

    """

    def __init__(self, dataset, n=2):
        super(NWays, self).__init__(dataset)
        self.n = n
        self.indices_to_labels = dict(dataset.indices_to_labels)

    def __reduce__(self):
        return NWays, (self.dataset, self.n)

    def new_task(self):  # Efficient initializer
        labels = self.dataset.labels
        task_description = []
        labels_to_indices = dict(self.dataset.labels_to_indices)
        # Randomly sample n classes from all possible classes (labels)
        if self.n > len(labels):
            raise ValueError(
                "Cannot sample {} classes from {} available classes.".format(
                    self.n, len(labels)))
        classes = random.sample(labels, k=self.n)
        for cl in classes:
            # Collect all images from these classes (labels)
            for idx in labels_to_indices[cl]:
                task_description.append(DataDescription(idx))
        return task_description

    def __call__(self, task_description):
        if task_description is None:
            return self.new_task()
        classes = []
        result = []
        set_classes = set()
        for dd in task_description:
            set_classes.add(self.indices_to_labels[dd.index])
        classes = random.sample(classes, k=self.n)
        for dd in task_description:
            if self.indices_to_labels[dd.index] in classes:
                result.append(dd)
        return result


class KShots(TaskTransform):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps K samples for each present labels.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **k** (int, *optional*, default=1) - The number of samples per label.
    * **replacement** (bool, *optional*, default=False) - Whether to sample with replacement.

    """

    def __init__(self, dataset, k: int=1, replacement: bool=False):
        super(KShots, self).__init__(dataset)
        self.dataset = dataset
        self.k = k
        self.replacement = replacement

    def __reduce__(self):
        return KShots, (self.dataset, self.k, self.replacement)

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        # TODO: The order of the data samples is not preserved.
        class_to_data = collections.defaultdict(list)

        # Take note of all class labels present in the current sample
        for dd in task_description:
            cls = self.dataset.indices_to_labels[dd.index]
            class_to_data[cls].append(dd)
        if self.replacement:
            def sampler(x, k):
                return [copy.deepcopy(dd)
                        for dd in random.choices(x, k=k)]
        else:
            sampler = random.sample

        try:
            # Sample 'k' datapoints from each class (label)
            return list(itertools.chain(*[sampler(dds, k=self.k) for dds in class_to_data.values()]))
        except ValueError:
            # Catch the case where there are not enough datapoints
            raise ValueError("Asked to sample {} datapoints from {} available datapoints.".format(
                self.k, [len(x) for x in class_to_data.values()]))


class RemapLabels(TaskTransform):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Given samples from K classes, maps the labels to 0, ..., K.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset, shuffle=True):
        super(RemapLabels, self).__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle

    def remap(self, data, mapping):
        data = [d for d in data]
        data[1] = mapping(data[1])
        return data

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        labels = list(set(self.dataset.indices_to_labels[dd.index] for dd in task_description))
        if self.shuffle:
            random.shuffle(labels)

        def mapping(x):
            return labels.index(x)

        for dd in task_description:
            remap = functools.partial(self.remap, mapping=mapping)
            dd.transforms.append(remap)
        
        # Fancy way to map given labels to [0, ..., K-1] range
        return task_description


class LoadData(TaskTransform):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Loads a sample from the dataset given its index.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.

    """

    def __init__(self, dataset):
        super(LoadData, self).__init__(dataset)
        self.dataset = dataset

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        # Fetch specified indices from the original dataset
        for data_description in task_description:
            data_description.transforms.append(lambda x: self.dataset[x])
        return task_description


class TaskDataset(Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/task_dataset.py)

    **Description**

    Creates a set of tasks from a given Dataset.

    In addition to the Dataset, TaskDataset accepts a list of task transformations (`task_transforms`)
    which define the kind of tasks sampled from the dataset.

    The tasks are lazily sampled upon indexing (or calling the `.sample()` method), and their
    descriptions cached for later use.
    If `num_tasks` is -1, the TaskDataset will not cache task descriptions and instead continuously resample
    new ones.
    In this case, the length of the TaskDataset is set to 1.

    For more information on tasks and task descriptions, please refer to the
    documentation of task transforms.

    **Arguments**

    * **dataset** (Dataset) - Dataset of data to compute tasks.
    * **task_transforms** (list, *optional*, default=None) - List of task transformations.
    * **num_tasks** (int, *optional*, default=-1) - Number of tasks to generate.

    **Example**
    ~~~python
    dataset = l2l.data.MetaDataset(MyDataset())
    transforms = [
        l2l.data.transforms.NWays(dataset, n=5),
        l2l.data.transforms.KShots(dataset, k=1),
        l2l.data.transforms.LoadData(dataset),
    ]
    taskset = TaskDataset(dataset, transforms, num_tasks=20000)
    for task in taskset:
        X, y = task
    ~~~
    """

    def __init__(self, dataset, task_transforms=None, num_tasks=-1, task_collate=None):
        if not isinstance(dataset, MetaDataset):
            raise ValueError("Please wrap your dataset in MetaDataset before passing on to TaskDataset.")

        if task_transforms is None:
            task_transforms = []
        if task_collate is None:
            task_collate = collate.default_collate
        if num_tasks < -1 or num_tasks == 0:
            raise ValueError('num_tasks needs to be -1 (infinity) or positive.')
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.task_transforms = task_transforms
        self.sampled_descriptions = {}  # Maps indices to tasks' description dict
        self.task_collate = task_collate
        self._task_id = 0

    def sample_task_description(self):
        #  Samples a new task description.
        #  list description = fast_allocate(len(self.dataset))
        description = None
        if callable(self.task_transforms):
            return self.task_transforms(description)
        for transform in self.task_transforms:
            description = transform(description)
        return description

    def get_task(self, task_description):
        # Given a task description, creates the corresponding batch of data.
        all_data = []
        for data_description in task_description:
            data = data_description.index
            # Apply transforms present in the specirfied data (if any)
            for transform in data_description.transforms:
                data = transform(data)
            all_data.append(data)
        return self.task_collate(all_data)

    def sample(self):
        """
        **Description**

        Randomly samples a task from the TaskDataset.

        **Example**
        ~~~python
        X, y = taskset.sample()
        ~~~
        """
        i = random.randint(0, len(self) - 1)
        return self[i]

    def __len__(self):
        if self.num_tasks == -1:
            # Ok to return 1, since __iter__ will run forever
            # and __getitem__ will always resample.
            return 1
        return self.num_tasks

    def __getitem__(self, i):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())
        if i not in self.sampled_descriptions:
            self.sampled_descriptions[i] = self.sample_task_description()
        task_description = self.sampled_descriptions[i]
        return self.get_task(task_description)

    def __iter__(self):
        self._task_id = 0
        return self

    def __next__(self):
        if self.num_tasks == -1:
            return self.get_task(self.sample_task_description())

        if self._task_id < self.num_tasks:
            task = self[self._task_id]
            self._task_id += 1
            return task
        else:
            raise StopIteration

    def __add__(self, other):
        msg = 'Adding datasets not yet supported for TaskDatasets.'
        raise NotImplementedError(msg)
