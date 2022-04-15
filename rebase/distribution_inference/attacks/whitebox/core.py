import torch as ch
import warnings
from torch.utils.data import Dataset

from distribution_inference.config import WhiteBoxAttackConfig, DatasetConfig
from distribution_inference.utils import warning_string


class Attack:
    def __init__(self,
                 config: WhiteBoxAttackConfig):
        self.config = config
        self.trained_model = False

    def _prepare_model(self):
        """
            Define and prepare model for attack.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def execute_attack(self, train_data, test_data, val_data=None, **kwargs):
        """
            Involves training meta-classifier, etc.
            After this method, attack should be ready to use.
            Return model performance (accuacy) on test data.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def save_model(self,
                   data_config: DatasetConfig,
                   attack_specific_info_string: str):
        """
            Save model to disk.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def eval_attack(self, test_loader, epochwise_version: bool = False):
        """
            Evaluate attack on given test data
        """
        if not self.trained_model:
            warnings.warn(warning_string("\nModel not trained/loaded, being used for eval\n"))
        return self._eval_attack(test_loader, epochwise_version)

    def _eval_attack(self, test_loader, epochwise_version: bool = False):
        raise NotImplementedError("Must be implemented in subclass")

    def load_model(self, path):
        self._prepare_model()
        self.model.load_state_dict(ch.load(path))
        self.trained_model = True


class BasicDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X = X
        self.Y = Y
        if self.Y is not None:
            assert len(self.X) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is None:
            return self.X[idx]
        return self.X[idx], self.Y[idx]
