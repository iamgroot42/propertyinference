from collections import OrderedDict
from typing import List, Tuple, Union
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch as ch
import gc

from distribution_inference.attacks.blackbox.per_point import PerPointThresholdAttack
from distribution_inference.attacks.blackbox.standard import LossAndThresholdAttack
from distribution_inference.attacks.blackbox.core import PredictionsOnOneDistribution
from distribution_inference.datasets.base import CustomDatasetWrapper
from distribution_inference.datasets.utils import (
    collect_gallery_images,
    get_match_scores,
)

from distribution_inference.attacks.blackbox.epoch_loss import Epoch_LossAttack
from distribution_inference.attacks.blackbox.epoch_threshold import (
    Epoch_ThresholdAttack,
)
from distribution_inference.attacks.blackbox.epoch_perpoint import Epoch_Perpoint
from distribution_inference.attacks.blackbox.epoch_meta import Epoch_Tree
from distribution_inference.attacks.blackbox.perpoint_choose import PerPointChooseAttack
from distribution_inference.attacks.blackbox.perpoint_choose_dif import (
    PerPointChooseDifAttack,
)
from distribution_inference.attacks.blackbox.KL import KLAttack
from distribution_inference.attacks.blackbox.generative import GenerativeAttack
from distribution_inference.attacks.blackbox.binary_perpoint import (
    BinaryPerPointThresholdAttack,
)
from distribution_inference.attacks.blackbox.KL_regression import KLRegression
from distribution_inference.attacks.blackbox.label_KL import label_only_KLAttack
from distribution_inference.attacks.blackbox.zhang import ZhangAttack
from distribution_inference.attacks.blackbox.att_ratio_attack import AttRatioAttack
from distribution_inference.attacks.blackbox.face_audit import FaceAuditAttack
from distribution_inference.datasets.base import (
    make_gallery_query_splits,
    _make_gallery_query_split,
)


ATTACK_MAPPING = {
    "threshold_perpoint": PerPointThresholdAttack,
    "loss_and_threshold": LossAndThresholdAttack,
    "single_update_loss": Epoch_LossAttack,
    "single_update_threshold": Epoch_ThresholdAttack,
    "single_update_perpoint": Epoch_Perpoint,
    "epoch_meta": Epoch_Tree,
    "perpoint_choose": PerPointChooseAttack,
    "perpoint_choose_dif": PerPointChooseDifAttack,
    "KL": KLAttack,
    "generative": GenerativeAttack,
    "binary_perpoint": BinaryPerPointThresholdAttack,
    "KL_regression": KLRegression,
    "label_KL": label_only_KLAttack,
    "zhang": ZhangAttack,
    "att_ratio": AttRatioAttack,
    "face_auditor": FaceAuditAttack,
}


def get_attack(attack_name: str):
    wrapper = ATTACK_MAPPING.get(attack_name, None)
    if not wrapper:
        raise NotImplementedError(f"Attack {attack_name} not implemented")
    return wrapper


def calculate_accuracies(
    data, labels, use_logit: bool = True, multi_class: bool = False
):
    """
    Function to compute model-wise average-accuracy on
    given data.
    """
    # Get predictions from each model (each model outputs logits)
    if multi_class:
        assert len(data.shape) == 3, "Multi-class data must be 3D"
        preds = np.argmax(data, axis=2).astype("int")
    else:
        assert len(data.shape) == 2, "Data should be 2D"
        if use_logit:
            preds = (data >= 0).astype("int")
        else:
            preds = (data >= 0.5).astype("int")

    # Repeat ground-truth (across models)
    expanded_gt = np.repeat(np.expand_dims(labels, axis=1), preds.shape[1], axis=1)

    return np.average(1.0 * (preds == expanded_gt), axis=0)


def get_graph_preds(
    ds,
    indices,
    models: List[nn.Module],
    verbose: bool = False,
    multi_class: bool = False,
    latent: int = None,
):
    """
    Get predictions for given graph models
    """
    X = ds.get_features()
    Y = ds.get_labels()

    predictions = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Generating Predictions")
    for model in iterator:
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        # Get model outputs/preds
        prediction = model(ds.g, X, latent=latent)[indices].detach().cpu().numpy()
        if latent != None and not multi_class:
            prediction = prediction[:, 0]
        predictions.append(prediction)

        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()

    predictions = np.stack(predictions, 0)
    gc.collect()
    ch.cuda.empty_cache()

    labels = Y[indices].cpu().numpy()[:, 0]
    return predictions, labels


def _collect_embeddings(model, data, batch_size: int):
    embds = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        embds.append(model(batch.cuda(), only_embedding=True).detach())
    return ch.cat(embds, 0)


@ch.no_grad()
def get_relation_preds(support_images, query_images, model: nn.Module):
    """
    Extract model output values as similarities between images from the support set (target person)
    and each query image, for each of the given models.
    """
    # samples here are query images
    # batches here are support images, with '0' referring to the person of interest
    # and are expected to be sorted, with the first one corresponding to the person of interest

    # 1. Collect sum embedding ("prototype") for each class based on embedding
    # 2. Use relation model to get "similarity" between each query image and each class prototype
    # 3. Use similarity to predict class of each query image

    # compute features for query images
    query_features = model(query_images, embedding_mode=True)
    n_queries = query_features.shape[0]

    # compute features for support people
    concatenated_relations = []
    for person_images in support_images:
        sample_features = model(person_images, embedding_mode=True)
        # Aggregate to get "prototype" embedding
        sample_features = ch.sum(sample_features, 0).unsqueeze(0)
        sample_features = sample_features.repeat(n_queries, 1, 1, 1)
        concatenated_relations.append(ch.cat((sample_features, query_features), 1))
    concatenated_relations = ch.stack(
        concatenated_relations, 0
    )  # shape: (n_way, n_queries, 2 * feat_1, feat_2, feat_3)
    concatenated_relations = ch.transpose(
        concatenated_relations, 0, 1
    )  # shape: (n_queries, n_way, 2 * feat_1, feat_2, feat_3)
    relation_pairs = concatenated_relations.reshape(
        -1,
        concatenated_relations.shape[2],
        concatenated_relations.shape[3],
        concatenated_relations.shape[4],
    )

    # Relations is of shape (n_queries, n_people)
    relations = model(relation_pairs, embedding_mode=False).view(n_queries, -1)
    # Can be thought of as "logits" if working with some attack here
    return relations.detach()


def get_vic_adv_preds_on_distr_relation_net(
    models_vic: Tuple[List[nn.Module], List[nn.Module]],
    models_adv: List[Tuple[nn.Module, List]],
    ds_obj: CustomDatasetWrapper,
    batch_size: int,
    use_similarities: bool = True,
):
    """
    ds_obj should correspond here to adversary data with subject of interested 'included'
    """
    ds_obj, ds_vic_1, ds_vic_2 = ds_obj

    # Get 'num_support' information from ds_obj
    num_support = ds_obj.relation_config.k_shot
    num_query = ds_obj.relation_config.num_query_test

    """
    # Get loader (containing image of person of interest)
    _, loader_adv = ds_obj.get_loaders(shuffle=True, batch_size=batch_size, primed_for_training=False)

    # Generating vectors for victim
    # Collect images for person of interest

    imgs = []
    for batch in loader_adv:
        prop_labels = batch[2]
        # Only append images where prop_labels is 1
        imgs.append(batch[0][prop_labels == 1])
    imgs = ch.cat(imgs, 0)
    split = _make_gallery_query_split(imgs, num_support)
    """

    relation_values_1, relation_values_2 = [], []
    """
    # Populate these values for models trained with and without subject of interest
    for model in tqdm(models_vic[0], desc="Generating predictions for victim (0)"):
        relations = get_relation_preds([split[0]], split[1], model).detach().cpu().numpy()
        if use_similarities:
            sims = image_similarities(split[0], split[1])
            sorting_order = np.argsort(sims)[-num_query:]
            relation_values_1.append(np.concatenate((relations[sorting_order, 0], sims[sorting_order])))
        else:
            relation_values_1.append(relations[:, 0])
    for model in tqdm(models_vic[1], desc="Generating predictions for victim (1)"):
        relations = get_relation_preds([split[0]], split[1], model).detach().cpu().numpy()
        if use_similarities:
            sims = image_similarities(split[0], split[1])
            sorting_order = np.argsort(sims)[-num_query:]
            relation_values_2.append(np.concatenate((relations[sorting_order, 0], sims[sorting_order])))
        else:
            relation_values_2.append(relations[:, 0])
    """
    # Generate "vectors" for adversary models
    # For this scenario, models_adv[0] is good enough
    preds_adv_1, preds_adv_2 = [], []  # for (non-members, members)
    for model, train_ids in tqdm(
        zip(models_adv[0], models_adv[1]),
        desc="Generating predictions for adversary",
        total=len(models_adv[0]),
    ):
        num_task = 100  # 10

        # Since we really only use num_task per split, can do it right here (to save compute in loader collection)
        train_ids_pick = np.random.choice(train_ids, num_task, replace=False)
        non_members = ds_obj.get_non_members(train_ids)
        non_members_pick = np.random.choice(non_members, num_task, replace=False)

        # Get loaders
        loader_members = ds_obj.get_specified_loader(
            train_ids_pick, shuffle=True, batch_size=batch_size
        )
        loader_non_members = ds_obj.get_specified_loader(
            non_members_pick, shuffle=True, batch_size=batch_size
        )
        # Get random gallery-query splts for each person
        splits_member = make_gallery_query_splits(
            loader_members,
            num_support=num_support,
            num_task=num_task,
            num_query=num_query,
        )
        splits_non_member = make_gallery_query_splits(
            loader_non_members,
            num_support=num_support,
            num_task=num_task,
            num_query=num_query,
        )
        # Get model outputs
        for split in splits_non_member:
            relations = (
                get_relation_preds([split[0]], split[1], model).detach().cpu().numpy()
            )
            if use_similarities:
                sims = image_similarities(split[0], split[1])
                sorting_order = np.argsort(sims)[-num_query:]
                preds_adv_1.append(
                    np.concatenate((relations[sorting_order, 0], sims[sorting_order]))
                )
            else:
                preds_adv_1.append(relations[:, 0])
        for split in splits_member:
            relations = (
                get_relation_preds([split[0]], split[1], model).detach().cpu().numpy()
            )
            if use_similarities:
                sims = image_similarities(split[0], split[1])
                sorting_order = np.argsort(sims)[-num_query:]
                preds_adv_2.append(
                    np.concatenate((relations[sorting_order, 0], sims[sorting_order]))
                )
            else:
                preds_adv_2.append(relations[:, 0])

    for model, train_ids in tqdm(
        zip(models_vic[0][0], models_vic[0][1][0]),
        desc="Generating predictions for victim",
        total=len(models_vic[0][0]),
    ):
        num_task = 100

        # Since we really only use num_task per split, can do it right here (to save compute in loader collection)
        train_ids_pick = np.random.choice(train_ids, num_task, replace=False)
        non_members = ds_vic_1.get_non_members(train_ids)
        non_members_pick = np.random.choice(non_members, num_task, replace=False)

        loader_members = ds_vic_1.get_specified_loader(
            train_ids_pick, shuffle=True, batch_size=batch_size
        )
        loader_non_members = ds_vic_1.get_specified_loader(
            non_members_pick, shuffle=True, batch_size=batch_size
        )
        splits_member = make_gallery_query_splits(
            loader_members,
            num_support=num_support,
            num_task=num_task,
            num_query=num_query,
        )
        splits_non_member = make_gallery_query_splits(
            loader_non_members,
            num_support=num_support,
            num_task=num_task,
            num_query=num_query,
        )
        for split in splits_non_member:
            relations = (
                get_relation_preds([split[0]], split[1], model).detach().cpu().numpy()
            )
            if use_similarities:
                sims = image_similarities(split[0], split[1])
                # Pick num_query most similar images
                sorting_order = np.argsort(sims)[-num_query:]
                relation_values_1.append(
                    np.concatenate((relations[sorting_order, 0], sims[sorting_order]))
                )
            else:
                relation_values_1.append(relations[:, 0])
        for split in splits_member:
            relations = (
                get_relation_preds([split[0]], split[1], model).detach().cpu().numpy()
            )
            if use_similarities:
                sims = image_similarities(split[0], split[1])
                sorting_order = np.argsort(sims)[-num_query:]
                relation_values_2.append(
                    np.concatenate((relations[sorting_order, 0], sims[sorting_order]))
                )
            else:
                relation_values_2.append(relations[:, 0])

    # preds_adv_1 are feature vectors for non-members
    # preds_adv_2 are feature vectors for members
    adv_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_adv_1, preds_property_2=preds_adv_2
    )

    # relation_values_1 are feature vectors for models that did not use subject of interest
    # relation_values_2 are feature vectors for models that did use subject of interest
    vic_preds = PredictionsOnOneDistribution(
        preds_property_1=relation_values_1, preds_property_2=relation_values_2
    )
    return adv_preds, vic_preds


def image_similarities(gallery, query):
    """
    Get similarity values between each query image and gallery image (mean)
    """
    similarities = []
    for gal_img in gallery:
        sims = ch.cosine_similarity(
            gal_img.view(1, -1) / 2 + 0.5,
            query.view(query.shape[0], -1) / 2 + 0.5,
            dim=1,
        )
        similarities.append(sims)
    similarities = ch.mean(ch.stack(similarities, 0), 0)
    return similarities.numpy()


def get_contrastive_preds(
    loader,
    gallery_data,
    models: List[nn.Module],
    preload: bool = False,
    verbose: bool = True,
    get_prop_labels: bool = False,
):
    """
    Get predictions for given models on given data.
    Valid for contrastive models only. Uses gallery images
    to compute probability distribution estimates for
    predictions on those gallery images.
    """
    predictions = []
    ground_truth = []
    prop_labels = []
    inputs = []
    batch_size = None
    # Accumulate all data for given loader
    for data in tqdm(loader, desc="Accumulating data"):
        if len(data) == 2:
            features, labels = data
            if get_prop_labels:
                raise ValueError("Loader does not return prop labels")
        else:
            features, labels, plabel = data
            if batch_size is None:
                batch_size = plabel.shape[0]
            if get_prop_labels:
                prop_labels.append(plabel.cpu().numpy())
        ground_truth.append(labels.cpu().numpy())
        if preload:
            inputs.append(features.cuda())
    ground_truth = np.concatenate(ground_truth, axis=0)
    if get_prop_labels:
        prop_labels = np.concatenate(prop_labels, axis=0)

    # Get predictions for each model
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Generating Predictions")
    for model in iterator:
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        # Get gallery embeddings
        # TODO: Too many images in gallery- batch it up to get embeddings
        gallery_embeddings = _collect_embeddings(
            model, gallery_data, batch_size=batch_size
        )

        with ch.no_grad():
            predictions_on_model = []

            # Skip multiple CPU-CUDA copy ops
            if preload:
                for data_batch in inputs:
                    embedding = model(data_batch, only_embedding=True).detach()
                    prediction = get_match_scores(embedding, gallery_embeddings)
                    predictions_on_model.append(prediction.cpu())
            else:
                # Iterate through data-loader
                for data in loader:
                    data_points, labels, _ = data
                    embedding = model(data_points.cuda(), only_embedding=True).detach()
                    prediction = get_match_scores(embedding, gallery_embeddings)
                    predictions_on_model.append(prediction.cpu())
        predictions_on_model = ch.cat(predictions_on_model).numpy()
        predictions.append(predictions_on_model)
        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()
    predictions = np.stack(predictions, 0)
    if preload:
        del inputs
    gc.collect()
    ch.cuda.empty_cache()

    if get_prop_labels:
        ground_truth = (ground_truth, prop_labels)
    return predictions, ground_truth


def get_preds(
    loader,
    models: List[nn.Module],
    preload: bool = False,
    verbose: bool = True,
    multi_class: bool = False,
    latent: int = None,
    get_prop_labels: bool = False,
):
    """
    Get predictions for given models on given data
    """
    # Check if models are graph-related
    if models[0].is_graph_model:
        return get_graph_preds(
            ds=loader[0],
            indices=loader[1],
            models=models,
            verbose=verbose,
            latent=latent,
            multi_class=multi_class,
        )

    predictions = []
    ground_truth = []
    prop_labels = []
    inputs = []
    # Accumulate all data for given loader
    for data in loader:
        if len(data) == 2:
            features, labels = data
            if get_prop_labels:
                raise ValueError("Loader does not return prop labels")
        else:
            features, labels, plabel = data
            if get_prop_labels:
                prop_labels.append(plabel.cpu().numpy())
        ground_truth.append(labels.cpu().numpy())
        if preload:
            inputs.append(features.cuda())
    ground_truth = np.concatenate(ground_truth, axis=0)
    if get_prop_labels:
        prop_labels = np.concatenate(prop_labels, axis=0)

    # Get predictions for each model
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Generating Predictions")
    for model in iterator:
        # Shift model to GPU
        model = model.cuda()
        # Make sure model is in evaluation mode
        model.eval()
        # Clear GPU cache
        ch.cuda.empty_cache()

        with ch.no_grad():
            predictions_on_model = []

            # Skip multiple CPU-CUDA copy ops
            if preload:
                for data_batch in inputs:
                    if latent != None:
                        prediction = model(data_batch, latent=latent).detach()
                    else:
                        prediction = model(data_batch).detach()

                        # If None for whatever reason, re-run
                        # Weird bug that pops in every now and then
                        # Was valid only for LR in Sklearn models- commenting out for now
                        # if prediction is None:
                        #     if latent != None:
                        #         prediction = model(data_batch, latent=latent).detach()
                        #     else:
                        #         prediction = model(data_batch).detach()

                        if not multi_class:
                            prediction = prediction[:, 0]
                    predictions_on_model.append(prediction.cpu())
            else:
                # Iterate through data-loader
                for data in loader:
                    data_points, labels, _ = data
                    # Get prediction
                    if latent != None:
                        prediction = model(data_points.cuda(), latent=latent).detach()
                    else:
                        prediction = model(data_points.cuda()).detach()
                        if not multi_class:
                            prediction = prediction[:, 0]
                    predictions_on_model.append(prediction)
        predictions_on_model = ch.cat(predictions_on_model).cpu().numpy()
        predictions.append(predictions_on_model)
        # Shift model back to CPU
        model = model.cpu()
        del model
        gc.collect()
        ch.cuda.empty_cache()
    predictions = np.stack(predictions, 0)
    if preload:
        del inputs
    gc.collect()
    ch.cuda.empty_cache()

    if get_prop_labels:
        ground_truth = (ground_truth, prop_labels)
    return predictions, ground_truth


def _get_preds_accross_epoch(
    models, loader, preload: bool = False, multi_class: bool = False
):
    preds = []
    for e in models:
        p, gt = get_preds(loader, e, preload, multi_class=multi_class)
        preds.append(p)

    return (np.array(preds), np.array(gt))


def get_preds_epoch_on_dis(
    models, loader, preload: bool = False, multi_class: bool = False
):
    preds1, gt = _get_preds_accross_epoch(models[0], loader, preload, multi_class)
    preds2, _ = _get_preds_accross_epoch(models[1], loader, preload, multi_class)
    preds_wrapped = [
        PredictionsOnOneDistribution(preds_property_1=p1, preds_property_2=p2)
        for p1, p2 in zip(preds1, preds2)
    ]
    return (preds_wrapped, gt)


def _get_preds_for_vic_and_adv(
    models_vic: List[nn.Module],
    models_adv: List[nn.Module],
    loader,
    epochwise_version: bool = False,
    preload: bool = False,
    multi_class: bool = False,
    get_prop_labels: bool = False,
    n_people: int = None,
    gallery_images: ch.Tensor = None,
):
    # Sklearn models do not support logits- take care of that
    use_prob_adv = models_adv[0].is_sklearn_model
    if epochwise_version:
        use_prob_vic = models_vic[0][0].is_sklearn_model
    else:
        use_prob_vic = models_vic[0].is_sklearn_model

    # Check if contrastive model
    # Also set not_use_logits to False, since scores will be
    # normalized
    are_contrastive_models = False
    if epochwise_version:
        if models_vic[0][0].is_contrastive_model:
            are_contrastive_models = True
            use_prob_adv = True
            use_prob_vic = True
    else:
        if models_vic[0].is_contrastive_model:
            are_contrastive_models = True
            use_prob_adv = True
            use_prob_vic = True

    # Check logic for using logits or probabilities
    not_using_logits = use_prob_adv or use_prob_vic

    if type(loader) == tuple:
        #  Same data is processed differently for vic/adcv
        loader_vic, loader_adv = loader
    else:
        if are_contrastive_models:
            raise ValueError(
                "Contrastive models require adversary loaders for gallery images. Please check code"
            )
        # Same loader
        loader_adv = loader
        loader_vic = loader

    def to_preds(x):
        exp = np.exp(x)
        return exp / (1 + exp)

    if are_contrastive_models and gallery_images is None:
        # Collect gallery images
        gallery_images = collect_gallery_images(loader_adv, n_classes=n_people)

    # Get predictions for adversary models and data
    if are_contrastive_models:
        preds_adv, ground_truth_repeat = get_contrastive_preds(
            loader_adv,
            gallery_images,
            models_adv,
            preload=preload,
            get_prop_labels=get_prop_labels,
        )
    else:
        preds_adv, ground_truth_repeat = get_preds(
            loader_adv,
            models_adv,
            preload=preload,
            multi_class=multi_class,
            get_prop_labels=get_prop_labels,
        )

    if get_prop_labels:
        ground_truth_repeat, prop_labels_repeat = ground_truth_repeat
    if not_using_logits and not use_prob_adv:
        preds_adv = to_preds(preds_adv)

    # Get predictions for victim models and data
    if epochwise_version:
        if are_contrastive_models:
            raise NotImplementedError(
                "Contrastive models are not supported for epoch-wise mode (will add support later)"
            )
        # Track predictions for each epoch
        preds_vic = []
        for models_inside_vic in tqdm(models_vic):
            preds_vic_inside, ground_truth = get_preds(
                loader_vic,
                models_inside_vic,
                preload=preload,
                verbose=False,
                multi_class=multi_class,
                get_prop_labels=get_prop_labels,
            )
            if get_prop_labels:
                ground_truth, prop_labels = ground_truth
            if not_using_logits and not use_prob_vic:
                preds_vic_inside = to_preds(preds_vic_inside)

            # In epoch-wise mode, we need prediction results
            # across epochs, not models
            preds_vic.append(preds_vic_inside)
    else:
        if are_contrastive_models:
            preds_vic, ground_truth = get_contrastive_preds(
                loader_vic,
                gallery_images,
                models_vic,
                preload=preload,
                get_prop_labels=get_prop_labels,
            )
        else:
            preds_vic, ground_truth = get_preds(
                loader_vic,
                models_vic,
                preload=preload,
                multi_class=multi_class,
                get_prop_labels=get_prop_labels,
            )
        if get_prop_labels:
            ground_truth, prop_labels = ground_truth
    assert np.all(ground_truth == ground_truth_repeat), "Val loader is shuffling data!"
    if get_prop_labels:
        assert np.all(
            prop_labels == prop_labels_repeat
        ), "Val loader is shuffling data!"
    if get_prop_labels:
        ground_truth = (ground_truth, prop_labels)

    if are_contrastive_models:
        return preds_vic, preds_adv, ground_truth, not_using_logits, gallery_images
    return preds_vic, preds_adv, ground_truth, not_using_logits


def get_vic_adv_preds_on_distr_seed(
    models_vic: Tuple[List[nn.Module], List[nn.Module]],
    models_adv: Tuple[List[nn.Module], List[nn.Module]],
    loader,
    epochwise_version: bool = False,
    preload: bool = False,
    multi_class: bool = False,
):
    preds_vic_1, preds_adv_1, ground_truth = _get_preds_for_vic_and_adv(
        models_vic[0],
        models_adv[0],
        loader,
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class,
    )
    # Get predictions for second set of models
    preds_vic_2, preds_adv_2, _ = _get_preds_for_vic_and_adv(
        models_vic[1],
        models_adv[1],
        loader,
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class,
    )
    adv_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_adv_1, preds_property_2=preds_adv_2
    )
    vic_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_vic_1, preds_property_2=preds_vic_2
    )
    return (adv_preds, vic_preds, ground_truth)


def get_vic_adv_preds_on_distr(
    models_vic: Tuple[List[nn.Module], List[nn.Module]],
    models_adv: Union[
        Tuple[List[nn.Module], List[nn.Module]], Tuple[List[nn.Module], List]
    ],
    ds_obj: CustomDatasetWrapper,
    batch_size: int,
    epochwise_version: bool = False,
    preload: bool = False,
    multi_class: bool = False,
    make_processed_version: bool = False,
    gallery_images: ch.Tensor = None,
):
    # Check if models are graph-related
    are_graph_models = False
    if epochwise_version:
        if models_vic[0][0][0].is_graph_model:
            are_graph_models = True
    else:
        # if models_vic[0][0].is_graph_model:
        #     are_graph_models = True
        if models_vic[0][0][0].is_graph_model:
            are_graph_models = True

    # Check if models are relation-net based
    are_relation_net_models = False
    if epochwise_version:
        if models_vic[0][0][0].is_relation_based:
            are_relation_net_models = True
    else:
        # if models_vic[0][0].is_relation_based:
        #     are_relation_net_models = True
        if models_vic[0][0][0].is_relation_based:
            are_relation_net_models = True
    if are_relation_net_models:
        return get_vic_adv_preds_on_distr_relation_net(
            models_vic, models_adv, ds_obj, batch_size
        )

    # Check if models are contrastive-learning based
    are_contrastive_models = False
    if epochwise_version:
        if models_vic[0][0][0].is_contrastive_model:
            are_contrastive_models = True
    else:
        if models_vic[0][0].is_contrastive_model:
            are_contrastive_models = True

    n_people = None
    if are_graph_models:
        # No concept of 'processed'
        data_ds, (_, test_idx) = ds_obj.get_loaders(batch_size=batch_size)
        loader_vic = (data_ds, test_idx)
        loader_adv = loader_vic
    else:
        loader_for_shape, loader_vic = ds_obj.get_loaders(batch_size=batch_size)
        adv_datum_shape = next(iter(loader_for_shape))[0].shape[1:]

        if make_processed_version:
            # Make version of DS for victim that processes data
            # before passing on
            adv_datum_shape = ds_obj.prepare_processed_data(loader_vic)
            loader_adv = ds_obj.get_processed_val_loader(batch_size=batch_size)
        else:
            # Get val data loader (should be same for all models, since get_loaders() gets new data for every call)
            loader_adv = loader_vic
            if are_contrastive_models:
                n_people = ds_obj.n_people

        # TODO: Use preload logic here to speed things even more

    # Get predictions for first set of models
    return_obj = _get_preds_for_vic_and_adv(
        models_vic[0],
        models_adv[0],
        (loader_vic, loader_adv),
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class,
        n_people=n_people,
        gallery_images=gallery_images,
    )
    if are_contrastive_models:
        (
            preds_vic_1,
            preds_adv_1,
            ground_truth,
            not_using_logits,
            gallery_images,
        ) = return_obj
    else:
        preds_vic_1, preds_adv_1, ground_truth, not_using_logits = return_obj
    # Get predictions for second set of models
    return_obj = _get_preds_for_vic_and_adv(
        models_vic[1],
        models_adv[1],
        (loader_vic, loader_adv),
        epochwise_version=epochwise_version,
        preload=preload,
        multi_class=multi_class,
        n_people=n_people,
        gallery_images=gallery_images,
    )
    if are_contrastive_models:
        preds_vic_2, preds_adv_2, _, _, _ = return_obj
    else:
        preds_vic_2, preds_adv_2, _, _ = return_obj
    adv_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_adv_1, preds_property_2=preds_adv_2
    )
    vic_preds = PredictionsOnOneDistribution(
        preds_property_1=preds_vic_1, preds_property_2=preds_vic_2
    )
    if are_contrastive_models:
        return adv_preds, vic_preds, ground_truth, not_using_logits, gallery_images
    return adv_preds, vic_preds, ground_truth, not_using_logits


def compute_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups):
    """Compute the key metrics"""
    from aif360.metrics import ClassificationMetric

    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * (
        classified_metric_pred.true_positive_rate()
        + classified_metric_pred.true_negative_rate()
    )
    metrics[
        "Statistical parity difference"
    ] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics[
        "Average odds difference"
    ] = classified_metric_pred.average_odds_difference()
    metrics[
        "Equal opportunity difference"
    ] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    metrics[
        "False discovery rate difference"
    ] = classified_metric_pred.false_discovery_rate_difference()
    metrics[
        "False discovery rate ratio"
    ] = classified_metric_pred.false_discovery_rate_ratio()
    metrics[
        "False omission rate difference"
    ] = classified_metric_pred.false_omission_rate_difference()
    metrics[
        "False omission rate ratio"
    ] = classified_metric_pred.false_omission_rate_ratio()
    metrics[
        "False negative rate difference"
    ] = classified_metric_pred.false_negative_rate_difference()
    metrics[
        "False negative rate ratio"
    ] = classified_metric_pred.false_negative_rate_ratio()
    metrics[
        "False positive rate difference"
    ] = classified_metric_pred.false_positive_rate_difference()
    metrics[
        "False positive rate ratio"
    ] = classified_metric_pred.false_positive_rate_ratio()

    return metrics
