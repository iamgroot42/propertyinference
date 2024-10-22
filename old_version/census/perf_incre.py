from tkinter import Y
from model_utils import get_models_path, load_model, BASE_MODELS_DIR
from data_utils import CensusTwo, CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_threshold_acc, find_threshold_acc, flash_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from perf_quart import get_models,get_pred,select_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratios for D_1")
    parser.add_argument('--ratio_2', help="ratios for D_2")
    parser.add_argument('--tries', type=int,
                        default=5, help="number of trials")
    args = parser.parse_args()
    flash_utils(args)

    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))
    avg_thre = []
    each_thre = []
    for _ in range(args.tries):
        thresholds= []
        # Load adv models
        total_models = 100
        models_1 = get_models(get_models_path(
            args.filter, "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(get_models_path(
            args.filter, "adv", args.ratio_2), total_models // 2)

        if args.filter == "two_attr":
            ds_1 = CensusTwo()
            ds_2 = CensusTwo()
            [r11,r12] = args.ratio_1.split(',')
            r11,r12 = float(r11),float(r12)
            [r21,r22] = args.ratio_2.split(',')
            r21,r22 = float(r21),float(r22)
            _, (x_te_1, y_te_1), _ = ds_1.get_data('adv',r11,r12)
            _, (x_te_2, y_te_2), _ = ds_2.get_data('adv',r21,r22)
        else:
            # Prepare data wrappers
            ds_1 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_1), split="adv")
            ds_2 = CensusWrapper(
                filter_prop=args.filter,
                ratio=float(args.ratio_2), split="adv")

        # Fetch test data from both ratios
            _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
            _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
        #y_te_1 = y_te_1.ravel()
        #y_te_2 = y_te_2.ravel()
        re1 = select_points(models_1,models_2,x_te_1,y_te_1)
        re2 = select_points(models_1,models_2,x_te_2,y_te_2)
        x1,y1 = re1[:,:-2], re1[:,-2]
        x2,y2 = re2[:,:-2], re2[:,-2]
        yg = (y1,y2)
        p1 = (get_pred(x1,models_1), get_pred(x2,models_1))
        p2 = (get_pred(x1,models_2), get_pred(x2,models_2))
        pv1 = (get_pred(x1,models_victim_1), get_pred(x2,models_victim_1))
        pv2 = (get_pred(x1,models_victim_2), get_pred(x2,models_victim_2))
        if (p1[0].shape[0] != x1.shape[0]):
            print('wrong dimension')
            break
        for i in tqdm(range(1,x1.shape[0]+1)):
            f_accs = []
            allaccs_1, allaccs_2 = [], []
            adv_accs = []
             #tr,rl = [],[]
            
            
            for j in range(2):
            #get accuracies
                
                
                accs_1 = np.average((p1[j][:i]==np.repeat(np.expand_dims(yg[j][:i],axis=1),p1[j].shape[1],axis=1)).astype(int),axis=0)
                accs_2 = np.average((p2[j][:i]==np.repeat(np.expand_dims(yg[j][:i],axis=1),p2[j].shape[1],axis=1)).astype(int),axis=0)

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
                accs_victim_1 = np.average((pv1[j][:i]==np.repeat(np.expand_dims(yg[j][:i],axis=1),pv1[j].shape[1],axis=1)).astype(int),axis=0)
                accs_victim_2 = np.average((pv2[j][:i]==np.repeat(np.expand_dims(yg[j][:i],axis=1),pv2[j].shape[1],axis=1)).astype(int),axis=0)

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

            # Collect all accuracies for basic baseline
                allaccs_1.append(accs_victim_1)
                allaccs_2.append(accs_victim_2)
        

        # Basic baseline: look at model performance on test sets from both G_b
        # Predict b for whichever b it is higher
            adv_accs = np.array(adv_accs)
            allaccs_1 = np.array(allaccs_1)
            allaccs_2 = np.array(allaccs_2)

            preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
            preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])

            basic_baseline_acc = (np.mean(preds_1) + np.mean(preds_2)) / 2
            #print("Basic baseline accuracy: %.3f" % (100 * basic_baseline_acc))

        # Threshold baseline: look at model performance on test sets from both G_b
        # and pick the better one
            #print("Threshold-test baseline accuracy: %.3f" %
              #(f_accs[np.argmax(adv_accs)]))

            
            thresholds.append(f_accs[np.argmax(adv_accs)])
        each_thre.append(thresholds)
    avg_thre = np.mean(np.array(each_thre),axis=0)
    x_lst = np.array(list(range(1,len(avg_thre)+1)))

       # tr = tr[np.argmax(adv_accs)]
       # rl = rl[np.argmax(adv_accs)]
    plt.plot(x_lst,avg_thre)
    plt.title('{}: {}vs{}'.format(args.filter,args.ratio_1,args.ratio_2))
    plt.ylim(30,101)
    plt.xlabel('number of points')
    plt.ylabel('average accuracy')
    plt.savefig('./images/perf_incre_{}_{}vs{}.png'.format(args.filter,args.ratio_1,args.ratio_2))
   
    
