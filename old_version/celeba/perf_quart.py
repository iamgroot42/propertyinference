from model_utils import get_models, BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES
import torch.nn as nn
import torch as ch
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from model_utils import get_model
mpl.rcParams['figure.dpi'] = 200


def get_preds(loader, ms):

    ps = []
    for model in tqdm(ms[1]):
        m = get_model(os.path.join(ms[0], model))
        m = nn.DataParallel(m.cuda(), device_ids=[3, 0, 1, 2])
        #m.cuda()
        m.eval()
        p = []
        ch.cuda.empty_cache()
        with ch.no_grad():
            for data in loader:
                images, _, _ = data
                images = images.cuda()
                #p.append(m(images).detach()[:,0].to(ch.device('cpu')).numpy())
                p.append(m(images).detach()[:, 0])
        p = ch.cat(p)
        #p = np.concatenate(p)

        ps.append(p)
        del m
    #ps = np.array(ps)
    ps = ch.stack(ps, 0)
    return ps.to(ch.device('cpu')).numpy()


def order_points(p1s, p2s):
    abs_dif = np.absolute(np.sum(p1s, axis=0)-np.sum(p2s, axis=0))
    inds = np.argsort(abs_dif)

    return inds


def cal_acc(p, y):
    outputs = (p >= 0).astype('int')
    return np.average((outputs == np.repeat(np.expand_dims(y, axis=1), outputs.shape[1], axis=1)).astype(int), axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--task', default="Smiling",
                        choices=SUPPORTED_PROPERTIES,
                        help='task to focus on')
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--total_models', type=int, default=100)
    parser.add_argument('--adv_adv_prefix', type=str,
                        default="adv_train",
                        help="Prefix for adversarial models for adv")
    parser.add_argument('--victim_adv_prefix', type=str,
                        default="adv_train",
                        help="Prefix for adversarial models for victim")
    parser.add_argument('--use_adv_for_adv', action="store_true",
                        help="Use adv-trained models for adv's models")
    parser.add_argument('--use_adv_for_victim', action="store_true",
                        help="Use adv-trained models for victim's models")
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    args = parser.parse_args()
    flash_utils(args)
    lst = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]  # ratio of data points to try
    ch.cuda.set_device(3)

    # Load victim models
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.ratio_1))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.ratio_2))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.ratio_1))
    test_dir_2 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.ratio_2))

    if args.use_adv_for_adv:
        print("Using adv-trained models for adv's models")
        train_dir_1 = os.path.join(train_dir_1, args.adv_adv_prefix)
        train_dir_2 = os.path.join(train_dir_2, args.adv_adv_prefix)

    if args.use_adv_for_victim:
        print("Using adv-trained models for victim's models")
        test_dir_1 = os.path.join(test_dir_1, args.victim_adv_prefix)
        test_dir_2 = os.path.join(test_dir_2, args.victim_adv_prefix)

    print("Loading models")
    models_victim_1 = get_models(test_dir_1)
    models_victim_2 = get_models(test_dir_2)

    # Load adv models
    total_models = args.total_models

    each_thre = []
    each_adv = []
    avg_thre = []
    ds_1 = CelebaWrapper(args.filter, float(
        args.ratio_1), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)
    ds_2 = CelebaWrapper(args.filter, float(
        args.ratio_2), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)

    # Get loaders
    loaders = [
        ds_1.get_loaders(args.batch_size, shuffle=False)[1],
        ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]
    ygs = []
    for i in range(2):
        yl = []
        for data in loaders[i]:
            _, y, _ = data
            yl.append(y.to(ch.device('cpu')).numpy())
        yl = np.concatenate(yl)
        ygs.append(yl)
    pvs1 = [get_preds(loaders[0], models_victim_1),
            get_preds(loaders[1], models_victim_1)]
    pvs2 = [get_preds(loaders[0], models_victim_2),
            get_preds(loaders[1], models_victim_2)]
    for _ in range(args.tries):
        thresholds = []
        adv_thresholds = []
        yg = [[], []]
        # Load adv models
        models_1 = get_models(train_dir_1, total_models // 2)
        models_2 = get_models(train_dir_2, total_models // 2)

        pv1, pv2 = [[], []], [[], []]
        p1 = [get_preds(loaders[0], models_1), get_preds(loaders[1], models_1)]
        p2 = [get_preds(loaders[0], models_2), get_preds(loaders[1], models_2)]

        ord = (order_points(p1[0], p2[0]), order_points(p1[1], p2[1]))
        for i in range(2):
            p1[i] = np.transpose(p1[i])[ord[i]][::-1]
            p2[i] = np.transpose(p2[i])[ord[i]][::-1]
            pv1[i] = np.transpose(pvs1[i])[ord[i]][::-1]
            pv2[i] = np.transpose(pvs2[i])[ord[i]][::-1]
            yg[i] = ygs[i][ord[i]][::-1]
        for ratio in lst:
            f_accs = []

            adv_accs = []
            #tr,rl = [],[]

            for j in range(2):
                #get accuracies

                leng = int(ratio*p1[j].shape[0])
                accs_1 = cal_acc(p1[j][:leng], yg[j][:leng])
                accs_2 = cal_acc(p2[j][:leng], yg[j][:leng])

            # Look at [0, 100]
                accs_1 *= 100
                accs_2 *= 100

                tracc, threshold, rule = find_threshold_acc(
                    # accs_1, accs_2, granularity=0.01)
                    accs_1, accs_2, granularity=0.005)
                adv_accs.append(100 * tracc)
           # tr.append(threshold)
           # rl.append(rule)
            # Compute accuracies on this data for victim
                accs_victim_1 = cal_acc(pv1[j][:leng], yg[j][:leng])
                accs_victim_2 = cal_acc(pv2[j][:leng], yg[j][:leng])

            # Look at [0, 100]
                accs_victim_1 *= 100
                accs_victim_2 *= 100

            # Threshold based on adv models
                combined = np.concatenate((accs_victim_1, accs_victim_2))
                classes = np.concatenate(
                    (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
                specific_acc = get_threshold_acc(
                    combined, classes, threshold, rule)

               # print("[Victim] Accuracy at specified threshold: %.2f" %
               #   (100 * specific_acc))
                f_accs.append(100 * specific_acc)

            ind = np.argmax(adv_accs)
            thresholds.append(f_accs[ind])
            adv_thresholds.append(adv_accs[ind])
        each_adv.append(adv_thresholds)
        each_thre.append(thresholds)
    each_adv = np.array(each_adv)
    each_thre = np.array(each_thre)
    avg_thre = np.mean(each_adv[:, :-1], axis=0)
    best = np.argmax(avg_thre)
    content = 'At {}, best thresholds accuracy: {}\nAt {}, thresholds accuracy: {}'.format(
        lst[best], each_thre[:, best], 1.0, each_thre[:, -1])
    print(content)
    log_path = os.path.join(
        './log', "perf_quart_{}:{}".format(args.filter, args.ratio_1))
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, args.ratio_2), "w") as wr:
        wr.write(content)
