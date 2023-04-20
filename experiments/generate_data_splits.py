from distribution_inference.datasets.utils import get_dataset_wrapper, get_dataset_information
from distribution_inference.config import DatasetConfig

di = get_dataset_information("celeba_person")()
# di.generate_victim_adversary_splits(adv_ratio=0.25, test_ratio=0.15)
di.generate_victim_adversary_splits(adv_ratio=0.2)
# ds = get_dataset_wrapper("maadface")
# x = ds(None)
# z = x.ds.get_data("victim", 1.0, "ethnicity")

# di = get_dataset_information("celeba")()
# di._extract_pretrained_features()

# ds = get_dataset_wrapper("celeba")
# config = DatasetConfig(name="celeba", prop="Male", value=0.5, split="adv", classify="Smiling", processed_variant=True)
# x = ds(config)
# _ , loader = x.get_loaders(batch_size=512)
# x, y, z = next(iter(loader))
# print(x.shape)