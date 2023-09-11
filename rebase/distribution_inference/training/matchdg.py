"""
    Our implementation of MatchDG: https://arxiv.org/abs/2006.07500
"""
from distribution_inference.config import TrainConfig
import torch as ch
import itertools
from tqdm import tqdm
import numpy as np

from distribution_inference.utils import warning_string, model_compile_supported


def cosine_sim(x, y):
    """
        Cosine similarity between vectors
    """
    cos = ch.nn.CosineSimilarity(dim=1, eps=1e-8)
    return cos(x, y)


def dist(x, y):
    """
        Euclidean distance between vectors
    """
    return ch.norm(x - y, dim=1)


def train(model, loaders, train_config: TrainConfig):
    # Get data loaders
    if len(loaders) == 2:
        train_loader, test_loader = loaders
        val_loader = None
        if train_config.get_best:
            print(warning_string("\nUsing test-data to pick best-performing model\n"))
    else:
        train_loader, test_loader, val_loader = loaders
    
    # if model_compile_supported():
    #    model = ch.compile(model)

    # Get match-DG hyper-parameters
    matchdg_config = train_config.misc_config.matchdg_config
    tau = matchdg_config.tau
    match_update_freq = matchdg_config.match_update_freq
    total_matches_per_point = matchdg_config.total_matches_per_point
    contrastive_epochs = matchdg_config.contrastive_epochs

    # Stage 1
    optim = ch.optim.SGD(model.parameters(),
                         lr=train_config.learning_rate,
                         weight_decay=train_config.weight_decay)
    for e in range(contrastive_epochs):
        updated_match_pairs(model, train_loader, train_config.batch_size, total_matches_per_point)
        tloss = train_epoch_contrastive(model, train_loader, optim, tau=tau, epoch=e+1, verbose=train_config.verbose)
        if (e + 1) % match_update_freq == 0:
            updated_match_pairs(model, train_loader, train_config.batch_size, total_matches_per_point)
    
    # Stage 2
    # Initialze new model from scratch
    with ch.no_grad():
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    # Re-initialize optimizer
    optim = ch.optim.SGD(model.parameters(),
                         lr=train_config.learning_rate,
                         weight_decay=train_config.weight_decay)

    for e in range(train_config.epochs):
        train_epoch_erm(model, train_loader, optim)



def loss_fn(model, batch, embeds_x, dataset, tau: float):
    """
        Contrastive loss, as described in (4) of paper.
    """
    pos_loss, neg_loss, tot_pairs = 0, 0, 0

    # Iterate across all matching pairs (same class, different domain)
    _, y, _, match_map, _ = batch
    for i in range(y.shape[0]):
        # Get matching data pairs
        match_pair_images = ch.stack([dataset[j][0] for j in match_map[i]], 0)
        match_pair_embeds = model.get_embedding(match_pair_images.cuda())
        
        pos_embeds = cosine_sim(embeds_x[i].unsqueeze(0), match_pair_embeds)
        pos_loss += ch.sum(pos_embeds) / tau

        # Non-matching pairs (different class) from within the batch
        non_matching_pair_indices = ch.where((y != y[i]))[0]
        neg_embeds = cosine_sim(embeds_x[i].unsqueeze(0), embeds_x[non_matching_pair_indices])
        neg_loss += ch.sum(ch.log(ch.sum(ch.exp(neg_embeds / tau)) + ch.exp(pos_embeds / tau)))
    
        tot_pairs += match_pair_embeds.shape[0]
    
    tot_loss = (neg_loss - pos_loss) / tot_pairs
    return tot_loss


@ch.no_grad()
def updated_match_pairs(model, loader, batch_size: int, total_matches_per_point: int):
    """
        Update match pairs, as described in Algorithm 1 of paper.
        For each anchor point, we want total_matches_per_point nearest same-class pairs.
    """
    dataset = loader.dataset

    y, embeds, ids = [], [], []
    for batch in loader:
        embeds.append(model.get_embedding(batch[0].cuda()))
        # embeds.append(model.get_embedding(batch[0].cuda()).cpu())
        y.append(batch[1])
        ids.append(batch[4])
    embeds = ch.cat(embeds, dim=0)
    y = ch.cat(y, dim=0)
    ids = ch.cat(ids, dim=0)
    # Identify embeddings for class 0, 1
    embeds_z = embeds[y == 0]
    embeds_o = embeds[y == 1]
    ids_z = ids[y == 0]
    ids_o = ids[y == 1]

    # For each y=0 data-point
    new_mapping_dict = {}
    for i, e in tqdm(enumerate(embeds_z), desc="Updating match pairs (y=0)", total=len(embeds_z)):
        # Get norm distances for these potential pairs
        candidate_distances = dist(e.unsqueeze(0), embeds_z)
        # Get indices of top total_matches_per_point nearest neighbors
        _, matching_pair_index = ch.topk(candidate_distances, total_matches_per_point, largest=False)
        matching_pair_index = matching_pair_index.cpu()
        # Exclude self from matching pairs
        matching_pair_index = matching_pair_index[matching_pair_index != i]
        # Set these as match pairs
        new_mapping_dict[ids_z[i].item()] = ids_z[matching_pair_index]

    # For each y=1 data-point
    for i, e in tqdm(enumerate(embeds_o), desc="Updating match pairs (y=1)", total=len(embeds_o)):
        # Get norm distances for these potential pairs
        candidate_distances = dist(e.unsqueeze(0), embeds_o)
        # Get indices of top total_matches_per_point nearest neighbors
        _, matching_pair_index = ch.topk(candidate_distances, total_matches_per_point, largest=False)
        matching_pair_index = matching_pair_index.cpu()
        # Exclude self from matching pairs
        matching_pair_index = matching_pair_index[matching_pair_index != i]
        # Set these as match pairs
        new_mapping_dict[ids_o[i].item()] = ids_o[matching_pair_index]

    # Before setting, make sure dictionary sizes match
    assert len(dataset.get_match_pairs_mapping()) == len(new_mapping_dict), "New mapping dict size does not match old mapping dict size"
    
    # Set on dataset
    dataset.set_match_pairs_mapping(new_mapping_dict)


def train_epoch_contrastive(model, loader, optim,
                            tau: float, epoch: int, verbose: bool = True):
    """
        Stage 1 of MatchDG- training feature extractor using contrastive loss
    """
    model.train()
    running_loss = 0.
    num_items = 0

    loader_ds = loader.dataset
    if verbose:
        iterator = tqdm(loader)
    else:
        iterator = loader

    for batch in iterator:
        embeds_x = model.get_embedding(batch[0].cuda())
        loss = loss_fn(model, batch, embeds_x, loader_ds, tau=tau)

        if verbose:
            iterator.set_description(
                '[Train] Epoch %d, Loss: %.5f' % (epoch, running_loss / num_items))

        # Backprop
        optim.zero_grad()
        loss.backward()
        optim.step()

        running_loss += loss.item()
        num_items += batch[0].shape[0]

    return running_loss / num_items


def train_epoch_erm(model, loader, optim):
    """
        Stage 2 of MatchDG- using inferred losses, and using a mix of
        inferred-match data and random-sample data. Contrastive+ERM loss on
        the first half, and just ERM on the second.
    """
    model.train()