from distribution_inference.utils import warning_string
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch as ch

from distribution_inference.datasets import new_census, celeba, boneage,census, texas, arxiv, synthetic, maadface, celeba_person, librispeech, maadface_person

DATASET_INFO_MAPPING = {
    "new_census": new_census.DatasetInformation,
    "celeba": celeba.DatasetInformation,
    "boneage": boneage.DatasetInformation,
    "old_census": census.DatasetInformation,
    "texas": texas.DatasetInformation,
    "arxiv": arxiv.DatasetInformation,
    "synthetic": synthetic.DatasetInformation,
    "maadface": maadface.DatasetInformation,
    "celeba_person": celeba_person.DatasetInformation,
    "librispeech": librispeech.DatasetInformation,
    "maadface_person": maadface_person.DatasetInformation
}

DATASET_WRAPPER_MAPPING = {
    "new_census": new_census.CensusWrapper,
    "celeba": celeba.CelebaWrapper,
    "boneage": boneage.BoneWrapper,
    "old_census": census.CensusWrapper,
    "texas": texas.TexasWrapper,
    "arxiv": arxiv.ArxivWrapper,
    "synthetic": synthetic.SyntheticWrapper,
    "maadface": maadface.MaadFaceWrapper,
    "celeba_person": celeba_person.CelebaPersonWrapper,
    "librispeech": librispeech.LibriSpeechWrapper,
    "maadface_person": maadface_person.MAADPersonWrapper
}


def get_dataset_wrapper(dataset_name: str):
    wrapper = DATASET_WRAPPER_MAPPING.get(dataset_name, None)
    if not wrapper:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return wrapper


def get_dataset_information(dataset_name: str):
    info = DATASET_INFO_MAPPING.get(dataset_name, None)
    if not info:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")
    return info


# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def filter(df, condition, ratio, verbose: bool = True, get_indices: bool = False):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            sampled_indices = np.concatenate((qualify, nqi))
            concat_df = pd.concat([df.iloc[qualify], df.iloc[nqi]])
            if get_indices:
                return concat_df, sampled_indices
            return concat_df
        
        if get_indices:
            return df.iloc[qualify], qualify
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            concat_df = pd.concat([df.iloc[qi], df.iloc[notqualify]])
            sampled_indices = np.concatenate((qi, notqualify))
            if get_indices:
                return concat_df, sampled_indices
            return concat_df

        if get_indices:
            return df.iloc[notqualify], notqualify
        return df.iloc[notqualify]


def heuristic(df, condition, ratio: float,
              cwise_sample: int,
              class_imbalance: float = 2.0,
              n_tries: int = 1000,
              tot_samples: int = None,
              class_col: str = "label",
              verbose: bool = True,
              get_indices: bool = False):
    if tot_samples is not None and class_imbalance is not None:
        raise ValueError("Cannot request class imbalance and total-sample based methods together")

    vals, pckds, indices = [], [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        # Binary class- simply sample (as requested)
        # From each class
        pckd_df, pckd_ids = filter(df, condition, ratio, verbose=False, get_indices=True)
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]
        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(
                    one_ids)[:cwise_sample]
            elif class_imbalance < 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(
                    one_ids)[:int(1 / class_imbalance * cwise_sample)]
            else:
                raise ValueError(f"Invalid class_imbalance value: {class_imbalance}")
            
            # Combine them together
            pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
            pckd_df = pckd_df.iloc[pckd]

        elif tot_samples is not None:
            # Combine both and randomly sample 'tot_samples' from them
            pckd = np.random.permutation(np.concatenate([zero_ids, one_ids]))[:tot_samples]
            pckd = np.sort(pckd)
            pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)
        indices.append(pckd_ids[pckd])

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    picked_indices = indices[np.argmin(vals)]
    if get_indices:
        return picked_df.reset_index(drop=True), picked_indices
    return picked_df.reset_index(drop=True)


def multiclass_heuristic(
        df, condition, ratio: float,
        total_samples: int,
        class_ratio_maintain: bool,
        n_tries: int = 1000,
        class_col: str = "label",
        verbose: bool = True):
    """
        Heuristic for ratio-based sampling, implemented
        for the multi-class setting.
    """
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)

    if class_ratio_maintain:
        class_labels, class_counts = np.unique(
            df[class_col].to_numpy(), return_counts=True)
        class_counts = class_counts / (1. * np.sum(class_counts))
        per_class_samples = class_counts * total_samples
    else:
        num_o = int(total_samples * ratio)
        num_z = total_samples - num_o
        # Binary class- simply sample (as requested)
        if ratio == 1:
            picked_df = df[condition(df)].sample(total_samples, replace=False)
        elif ratio == 0:
            picked_df = df[~condition(df)].sample(total_samples, replace=False)
        else:
            picked_o = df[condition(df)].sample(num_o, replace=False)
            picked_z = df[~condition(df)].sample(num_z, replace=False)
            # Combined these two
            picked_df = pd.concat([picked_o, picked_z])
        return picked_df.reset_index(drop=True)

    for _ in iterator:

        # For each class
        inner_pckds = []
        for i, cid in enumerate(class_labels):
            # Find rows that have that specific class label
            df_i = df[df[class_col] == cid]
            pcked_df = filter(df_i, condition, ratio, verbose=False)
            # Randomly sample from this set
            # Since sampling is uniform at random, should preserve ratio
            # Either way- we pick a sample that is closest to desired ratio
            # So that aspect should be covered anyway

            if int(per_class_samples[i]) < 1:
                raise ValueError(f"Not enough data to sample from class {cid}")
            if int(per_class_samples[i]) > len(pcked_df):
                print(warning_string(
                    f"Requested {int(per_class_samples[i])} but only {len(pcked_df)} avaiable for class {cid}"))
            else:
                pcked_df = pcked_df.sample(
                    int(per_class_samples[i]), replace=True)
            inner_pckds.append(pcked_df.reset_index(drop=True))
        # Concatenate all inner_pckds into one
        pckd_df = pd.concat(inner_pckds)

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)


def collect_data(loader, expect_extra: bool = True):
    X, Y = [], []
    for datum in loader:
        if expect_extra:
            x, y, _ = datum
        else:
            x, y = datum
        X.append(x)
        Y.append(y)
    # Concatenate both torch tensors across batches
    X = ch.cat(X, dim=0)
    Y = ch.cat(Y, dim=0)
    return X, Y


def collect_gallery_images(loader, n_classes):
    """
        Collect one image (to be used as gallery) per person
    """
    images = None
    collected = np.zeros(n_classes, dtype=np.bool)
    with tqdm(total=n_classes, desc="Collecting gallery images") as pbar:
        for x, y, _ in loader:
            data_shape = x.shape[1:]
            # Initialize images placeholder if not done so already
            if images is None:
                images = ch.zeros((n_classes, *data_shape), dtype=x.dtype)
            # New people = (not seen) AND (current batch)
            current_batch = np.zeros(n_classes, dtype=np.bool)
            current_batch[y] = True
            new_people = np.logical_and(~collected, current_batch)
            
            # Select people that we want images for
            new_people_ids = np.where(new_people)[0]
            if len(new_people_ids) == 0:
                continue

            wanted_ids = np.isin(y, new_people_ids)
            # Pick unique images out of these people
            unique_people, unique_indices = np.unique(y[wanted_ids], return_index=True)
            images[unique_people] = x[np.nonzero(wanted_ids)[0][unique_indices]]
            collected[unique_people] = True

            pbar.update(len(unique_people))

            # Stop if all collected
            if collected.all():
                break
    if not collected.all():
        raise ValueError("Not enough images per class")
    return images


def get_match_scores(batch_embeds, gallery_embeds, apply_softmax: bool = True):
    """
        Compute match scores (probabilities) via softmax-based normalization
        of cosine similarity products between batch_embeds and gallery_embeds.
    """
    # Compute cosine similarity product (normalized with norms) between both each image in batch and all gallery_embeds
    # batch_embeds is of shape (batch_size, embed_dim)
    # gallery_embeds is of shape (n_gallery, embed_dim)
    simulated_preds = []
    for b in batch_embeds:
         csim = ch.nn.functional.cosine_similarity(b, gallery_embeds)
         simulated_preds.append(csim)
    simulated_preds = ch.stack(simulated_preds, dim=0)
    if apply_softmax:
        # Normalize with softmax
        simulated_preds = ch.softmax(simulated_preds, dim=1)
    return simulated_preds
