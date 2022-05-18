"""
    Analyze overlap in model predictions using various meta-classification methods.
    Useful tp compare different methods and predict utility in trying to
    combine them.
"""
import utils
import numpy as np
from data_utils import BoneWrapper
from model_utils import get_model_folder_path, get_model_features, get_models
import argparse
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import torch as ch


def get_preds(loader, models):
    ps = []
    inp = []
    for data in loader:
        images, _, _ = data
        inp.append(images.cuda())

    for model in tqdm(models):
        # Shift model to CUDA and make sure it's in eval mode
        model = model.cuda()
        model.eval()
        p=[]
        ch.cuda.empty_cache()
        with ch.no_grad():
            for images in inp:
                p.append(model(images).detach()[:, 0])
        p = ch.cat(p)
        
        ps.append(p)
    ps = ch.stack(ps,0)
    return ps.to(ch.device('cpu')).numpy()


def get_best_blackbox_results(adv_1, adv_2, vic_1, vic_2, args):
    """
        Launch best black-box attack for this dataset and
        return accuracy, along with model-wise predictions.
    """
    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    df_1 = utils.heuristic(
        df_val, filter, float(args.d_0),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    df_2 = utils.heuristic(
        df_val, filter, float(args.d_1),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    # Prepare data loaders
    ds_1 = BoneWrapper(
        df_1, df_1, features=features)
    ds_2 = BoneWrapper(
        df_2, df_2, features=features)
    loaders = [
        ds_1.get_loaders(args.batch_size, shuffle=False)[1],
        ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]

    # Get predictions on adversary's models
    preds_1 = [get_preds(loaders[0], adv_1), get_preds(loaders[0], adv_2)]
    preds_2 = [get_preds(loaders[1], adv_1), get_preds(loaders[1], adv_2)]
    # Get predictions on victim's models
    preds_vic_1 = [get_preds(loaders[0], vic_1), get_preds(loaders[0], vic_2)]
    preds_vic_2 = [get_preds(loaders[1], vic_1), get_preds(loaders[1], vic_2)]

    # Get predictions using perpoint-threshold test
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4]
    (vic_acc, vic_preds), (adv_acc, adv_preds), _ = utils.perpoint_threshold_test(
        (preds_1, preds_2),
        (preds_vic_1, preds_vic_2),
        ratios, granularity=0.005)

    if not args.train_meta:
        vic_preds = (vic_preds >= 0.5)  # Treat as usual 0/1 predictions

    # Black-box preds aim for first half as 0s, we aim for other direction
    # So flip predictions
    vic_preds = 1. - vic_preds
    adv_preds = 1. - adv_preds

    return (vic_preds, vic_acc), (adv_preds, adv_acc)


def get_best_whitebox_results(dims, vic_1, vic_2, adv_1, adv_2, args):
    """
        Launch best white-box attack for this dataset and
        return accuracy, along with model-wise predictions.
    """
    preds_adv = None
    # Load meta-classifier
    metamodel = utils.PermInvModel(dims, dropout=0.5)
    # Load up state dict
    metamodel.load_state_dict(ch.load(args.meta))
    metamodel = metamodel.cuda()
    metamodel.eval()
    # Batch up data
    X = np.concatenate((vic_1, vic_2))
    X = utils.prepare_batched_data(X)
    # Get predictions on victim's models
    preds_vic = utils.get_meta_preds(metamodel, X, args.batch_size)

    if args.train_meta:
        X = np.concatenate((adv_1, adv_2))
        X = utils.prepare_batched_data(X)
        # Get predictions on adv's models
        preds_adv = utils.get_meta_preds(metamodel, X, args.batch_size)

        # Convert to probabilities
        preds_adv = ch.sigmoid(preds_adv)
        preds_vic = ch.sigmoid(preds_vic)

        preds_adv = preds_adv.cpu().numpy()[:, 0]
    else:
        preds_vic = 1. * (preds_vic >= 0)

    preds_vic = preds_vic.cpu().numpy()[:, 0]

    return preds_vic, preds_adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', required=True, help='path to meta-classifier')
    parser.add_argument('--batch_size', type=int,
                        default=512,
                        help='batch-size for meta-classifier inference')
    parser.add_argument('--d_0', default="0.5",
                        help='ratios to use for D_0')
    parser.add_argument('--d_1', default=None, help='target ratio')
    parser.add_argument('--train_meta', action="store_true",
                        help='train a meta-classifier on top of predictions from these two methods')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Load up test data with their features
    dims, pos_w_test, pos_models_vic = get_model_features(
        get_model_folder_path("victim", args.d_0), fetch_models=True,
        shuffle=False)
    _, neg_w_test, neg_models_vic = get_model_features(
        get_model_folder_path("victim", args.d_1), fetch_models=True,
        shuffle=False)

    # Load up "train" data for Threshold Test
    total_models = 100
    _, pos_models_adv = get_models(get_model_folder_path("adv", value=args.d_0),
                                n_models=total_models // 2, shuffle=True)
    _, neg_models_adv = get_models(get_model_folder_path("adv", value=args.d_1),
                                n_models=total_models // 2, shuffle=True)

    # Get black-box test predictions
    print("Getting predictions for black-box test")
    black_vic, black_adv = get_best_blackbox_results(
                                pos_models_adv, neg_models_adv,
                                pos_models_vic, neg_models_vic,
                                args)
    black_vic_preds = black_vic[0]
    black_adv_preds = black_adv[0]

    pos_w_train, neg_w_train = None, None
    if args.train_meta:
        # Extract model features from given white-box models
        _, pos_w_train  = get_model_features(
            pos_models_adv, fetch_models=False,
            shuffle=False, models_provided=True)
        _, neg_w_train = get_model_features(
            neg_models_adv, fetch_models=False,
            shuffle=False, models_provided=True)

    # Get meta-classifier predictions
    print("Getting predictions for white-box test")
    white_vic_preds, white_adv_preds = get_best_whitebox_results(
        dims, pos_w_test, neg_w_test, pos_w_train, neg_w_train, args)

    ground_truth_vic = [0.] * len(pos_w_test) + [1.] * len(neg_w_test)
    ground_truth_vic = np.array(ground_truth_vic)

    if args.train_meta:
        # Combine predictions from WB and BB models into 2-D feature vectors for train and test
        X_train = np.stack((white_adv_preds, black_adv_preds), 1)
        X_test = np.stack((white_vic_preds, black_vic_preds), 1)

        # Get white-box predictions for adversary's models
        ground_truth_adv = [0.] * len(pos_models_adv) + [1.] * len(neg_models_adv)
        ground_truth_adv = np.array(ground_truth_adv)

        # Train meta-classifier
        meta_meta_clf = RandomForestClassifier(
            max_depth=1, n_estimators=100, max_samples=0.4)
        meta_meta_clf.fit(X_train, ground_truth_adv)
        # Print train score
        print("Train score: {:.4f}".format(
            meta_meta_clf.score(X_train, ground_truth_adv)))
        # Print test score
        print("Test score: {:.4f}\n".format(
            meta_meta_clf.score(X_test, ground_truth_vic)))

        white_vic_preds = (white_vic_preds >= 0.5)
        black_vic_preds = (black_vic_preds >= 0.5)

    white_right = (white_vic_preds == ground_truth_vic)
    black_right = (black_vic_preds == ground_truth_vic)

    white_wrong_black_right = np.sum(np.logical_not(white_right) & black_right)
    white_right_black_wrong = np.sum(white_right & np.logical_not(black_right))
    white_right_black_right = np.sum(
        (white_vic_preds == ground_truth_vic) & (black_vic_preds == ground_truth_vic))
    max_together = np.sum(white_right) + \
        np.sum(black_right) - white_right_black_right

    # Combine accuracy if "only when both agree" models is followed
    when_both_agree = (white_vic_preds == black_vic_preds)
    both_agree_acc = np.mean(
        white_vic_preds[when_both_agree] == ground_truth_vic[when_both_agree])
    print("Accuracy when both agree: {:.4f}".format(both_agree_acc))
    print("Rejection rate (%) for this scenario: {:.2f}".format(
          100 * (1 - np.mean(when_both_agree))))

    # Compute upper bound (based on current accuracies) on using both methods
    print("White-box accuracy", np.mean(ground_truth_vic == white_vic_preds))
    print("Black-box accuracy", np.mean(ground_truth_vic == black_vic_preds))
    print("Maximim possible accuracy (%) on combining both",
          100 * max_together / len(ground_truth_vic))

    # Absolute gain in said combined method
    gain = (max_together - max(np.sum(white_right),
                               np.sum(black_right))) / len(ground_truth_vic)
    print("Potential accuracy gain (%) in combining both accuracies: {:.2f}".format(
        100 * gain))