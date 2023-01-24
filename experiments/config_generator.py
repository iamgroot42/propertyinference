"""
    Generate new synthetic data configurations, based on ones already explored.
"""
from distribution_inference.config import SyntheticDatasetConfig
import os
import dataclasses
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    # Specify parameter and its value that needs to be added to config
    parameter = 'dist_diff_mean'
    value = 0.5
    # Construct pandas object from records
    df = []
    start_with = 1 + max([int(x.split(".json")[0]) for x in os.listdir("configs/synthetic/data_configs/")])
    for file in os.listdir("configs/synthetic/data_configs/"):
        config_present = SyntheticDatasetConfig.load(f"configs/synthetic/data_configs/{file}")
        config_as_dict = dataclasses.asdict(config_present)
        df.append(config_as_dict)
    df = pd.DataFrame.from_dict(df)

    # Remove columns that do not have any variance in values
    df.loc[:, 'layer'] = df['layer'].apply(lambda x: ','.join([str(i) for i in x]))

    # Desired attribute configuration
    df[parameter] = value
    df = df.drop_duplicates()

    # Convert back to list
    df.loc[:, 'layer'] = df['layer'].apply(lambda x: ([int(i) for i in x.split(',')]))

    # Parse records
    for i, record in tqdm(enumerate(df.to_dict('records')), total=len(df)):
        config = SyntheticDatasetConfig.from_dict(record)
        savenum = start_with + i
        config.save(f"configs/synthetic/data_configs/{savenum}.json")
