import utils
from data_utils import SUPPORTED_PROPERTIES
from model_utils import get_models_path, get_model_representations, BASE_MODELS_DIR, save_model
import argparse
import numpy as np
import torch as ch
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def epoch_strategy(tg, args):
    return args.epochs
    # if args.filter == "race":
    #     return args.epochs if tg not in ["0.6", "0.7", "0.8"] else 70
    # else:
    #     return args.epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample', type=int, default=800,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=1000)
    # Sex: 1000 epochs, 1e-3
    # Race: 500* epochs, 1e-3
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Number of epochs to train meta-classifier")
    parser.add_argument('--start_n', type=int, default=0,
                        help="Only consider starting from this layer")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Use only first N layers' parameters")
    parser.add_argument('--ntimes', type=int, default=10,
                        help='number of repetitions for multimode')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--d_0', default="0.5", help='ratios to use for D_0')
    parser.add_argument('--trg', default=None, help='target ratios')
    parser.add_argument('--save', action="store_false", help='save model or not')
    parser.add_argument('--drop', action="store_true")
    parser.add_argument('--scale',type=float,default=1.0)
    parser.add_argument('--dp',type=float,default=None)
    parser.add_argument('--b',action="store_true")
    args = parser.parse_args()
    utils.flash_utils(args)
    adp=None
    if args.b:
        adp="DP_%.2f" %args.dp
    dp = None
    if args.dp:
        dp = "DP_%.2f" %args.dp
    d_0 = args.d_0
    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    # Only look at multiples of 0.10
    # targets = filter(lambda x: x != d_0 and int(float(x) * 10) ==
    #                 float(x) * 10, os.listdir(get_models_path(args.filter, "adv")))
    if args.trg == None:
        targets = sorted(['0.0','0.1','0.2','0.3','0.4','0.6','0.7','0.8','0.9','1.0'])
    else:
        lst = eval(args.trg)
        targets = []
        for i in lst:
            if type(i) is list:
                i = [str(x) for x in i]
                targets.append(','.join(i))

            else:
                targets.append(str(i))
        targets = sorted(targets)

    # targets = sorted(list(targets))

    # Load up positive-label test, test data
    if args.drop:
        pos_w, pos_labels, _ = get_model_representations(
        os.path.join(get_models_path(args.filter, "adv", d_0),'drop'), 1, args.first_n)
        pos_w_test, pos_labels_test, dims = get_model_representations(
        os.path.join(get_models_path(args.filter, "victim", d_0),'drop'), 1, args.first_n)
    else:
        if args.scale != 1:
            pos_w, pos_labels, _ = get_model_representations(
            os.path.join(get_models_path(args.filter, "adv", d_0),'sample_size_scale:{}'.format(args.scale)), 1, args.first_n)
            pos_w_test, pos_labels_test, dims = get_model_representations(
            os.path.join(get_models_path(args.filter, "victim", d_0),'sample_size_scale:{}'.format(args.scale)), 1, args.first_n)

        else:

            pos_w, pos_labels, _ = get_model_representations(
            get_models_path(args.filter, "adv", d_0,adp), 1, args.first_n)
            if args.dp:
                pos_w_test, pos_labels_test, dims = get_model_representations(
                get_models_path(args.filter, "victim", d_0,dp), 1, args.first_n)
            else:
                pos_w_test, pos_labels_test, dims = get_model_representations(
                get_models_path(args.filter, "victim", d_0), 1, args.first_n)

    data = []
    for tg in targets:
        tgt_data = []
        # Load up negative-label train, test data
        if args.drop:

            neg_w, neg_labels, _ = get_model_representations(
                os.path.join(get_models_path(args.filter, "adv", tg),'drop'), 0, args.first_n)
            neg_w_test, neg_labels_test, _ = get_model_representations(
                os.path.join(get_models_path(args.filter, "victim", tg),'drop'), 0, args.first_n)
        else:
            if args.scale != 1:
                neg_w, neg_labels, _ = get_model_representations(
                os.path.join(get_models_path(args.filter, "adv", tg),'sample_size_scale:{}'.format(args.scale)), 0, args.first_n)
                neg_w_test, neg_labels_test, dims = get_model_representations(
                os.path.join(get_models_path(args.filter, "victim", tg),'sample_size_scale:{}'.format(args.scale)), 0, args.first_n)
        
            else:

                neg_w, neg_labels, _ = get_model_representations(
                get_models_path(args.filter, "adv", tg,adp), 0, args.first_n)
                if args.dp:
                    neg_w_test, neg_labels_test, dims = get_model_representations(
                    get_models_path(args.filter, "victim", tg,dp), 0, args.first_n)
                else:
                    neg_w_test, neg_labels_test, dims = get_model_representations(
                    get_models_path(args.filter, "victim", tg), 0, args.first_n)

        pos_w = np.array(pos_w,dtype='object')
        pos_w_test = np.array(pos_w_test,dtype='object')
        neg_w = np.array(neg_w,dtype='object')
        neg_w_test = np.array(neg_w_test,dtype='object')
        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))
        Y_te = ch.cat((pos_labels_test, neg_labels_test)).cuda()

        print("Batching data: hold on")
        X_te = utils.prepare_batched_data(X_te)

        for i in range(args.ntimes):
            # Random shuffles
            shuffled_1 = np.random.permutation(len(pos_labels))
            pp_x = pos_w[shuffled_1[:args.train_sample]]
            pp_y = pos_labels[shuffled_1[:args.train_sample]]

            shuffled_2 = np.random.permutation(len(neg_labels))
            np_x = neg_w[shuffled_2[:args.train_sample]]
            np_y = neg_labels[shuffled_2[:args.train_sample]]

            # Combine them together
            X_tr = np.concatenate((pp_x, np_x))
            Y_tr = ch.cat((pp_y, np_y))

            val_data = None
            if args.val_sample > 0:
                pp_val_x = pos_w[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_x = neg_w[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                pp_val_y = pos_labels[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_y = neg_labels[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                # Combine them together
                X_val = np.concatenate((pp_val_x, np_val_x))
                Y_val = ch.cat((pp_val_y, np_val_y))

                # Batch layer-wise inputs
                print("Batching data: hold on")
                X_val = utils.prepare_batched_data(X_val)
                Y_val = Y_val.float()

                val_data = (X_val, Y_val)

            metamodel = utils.PermInvModel(dims, dropout=0.5)
            metamodel = metamodel.cuda()
            metamodel = ch.nn.DataParallel(metamodel)

            # Float data
            Y_tr = Y_tr.float()
            Y_te = Y_te.float()

            # Batch layer-wise inputs
            print("Batching data: hold on")
            X_tr = utils.prepare_batched_data(X_tr)

            # Train PIM
            clf, tacc = utils.train_meta_model(
                         metamodel,
                         (X_tr, Y_tr), (X_te, Y_te),
                         epochs=epoch_strategy(tg, args),
                         binary=True, lr=1e-3,
                         regression=False,
                         batch_size=args.batch_size,
                         val_data=val_data, combined=True,
                         eval_every=10, gpu=True)
            if args.save:
                if args.dp:
                    save_path = os.path.join(BASE_MODELS_DIR, args.filter,"DP_%.2f" % args.dp, "meta_model", "-".join(
                    [args.d_0, str(args.start_n), str(args.first_n)]), tg)
                else:
                    save_path = os.path.join(BASE_MODELS_DIR, args.filter,"DP_%.2f" % args.dp, "meta_model", "-".join(
                    [args.d_0, str(args.start_n), str(args.first_n)]), tg)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                save_model(clf, os.path.join(save_path, str(i)+
            "_%.2f" % tacc))
            tgt_data.append(tacc)
            print("Test accuracy: %.3f" % tacc)
        data.append(tgt_data)

    # Print data
    log_path = os.path.join(BASE_MODELS_DIR, args.filter, "meta_result")
    if args.b:
        log_path = os.path.join(log_path,"both_dp")
    if args.dp:
        l=os.path.join(log_path,dp)
    if args.scale != 1.0:
        log_path = os.path.join(log_path,"sample_size_scale:{}".format(args.scale))

    if args.drop:
        log_path = os.path.join(log_path,'drop')
    utils.ensure_dir_exists(log_path)
    with open(os.path.join(log_path, "-".join([args.filter, args.d_0, str(args.start_n), str(args.first_n)])), "a") as wr:
        for i, tup in enumerate(data):
            print(targets[i], tup)
            wr.write(targets[i]+': '+",".join([str(x) for x in tup])+"\n")
