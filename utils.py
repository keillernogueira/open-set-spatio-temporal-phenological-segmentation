import os
import argparse
import imageio
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, cohen_kappa_score
from collections import defaultdict

import torch


class TwoWayDict(dict):
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2


cedro = TwoWayDict({0: (0, 255, 255),  # cyan / classes train 0 and test 4
                    1: (0, 255, 0),  # green / classes train 1 and test 5
                    2: (0, 0, 255),  # blue / classes train 2 and test 6
                    3: (243,73,211),  # magenta / classes train 3 and test 7
                    4: (0, 0, 0),  # background / class 8
                    5: (255, 0, 0),  # red / openset
                    })  


def create_lookup_class(num_classes, hidden_class):
    lookup_class = np.zeros((num_classes+2, 3), dtype=np.uint8)  # +2 for background and openset
    count = 0
    for i in range(num_classes):
        if i == hidden_class:
            continue
        lookup_class[count] = cedro[i]
        count += 1
    lookup_class[num_classes+1] = (255, 0, 0)  # openset
    # print('lookup class', lookup_class)
    return lookup_class


def convert_pred_image_to_rgb(image, output_name):
    h, w = image.shape
    output = np.empty((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            output[i, j, :] = cedro[image[i, j]]
    imageio.imwrite(output_name, output)


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, acc, acc_cls, cm, num_saves=5):
    if len(best_records) < num_saves:
        best_records.append({'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i in range(len(best_records)):
            if best_records[min_index]['acc_cls'] > best_records[i]['acc_cls']:
                min_index = i
        # check if currect acc is greater than min saved acc
        if acc_cls > best_records[min_index]['acc_cls']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    np.save(os.path.join(output_path, 'best_records.npy'), best_records)


def evaluate_map(true_map, pred_map, num_classes, hidden_class, open_set_class=None):
    all_labels = None
    all_preds = None
    for c in range(num_classes-1):
        if all_labels is None:
            all_labels = true_map[true_map == c]
            all_preds = pred_map[true_map == c]
        else:
            all_labels = np.concatenate((all_labels, true_map[true_map == c]))
            all_preds = np.concatenate((all_preds, pred_map[true_map == c]))
    # print(np.asarray(all_labels).shape, np.asarray(all_preds).shape,
    #       np.bincount(np.asarray(all_labels).astype(int)), np.bincount(np.asarray(all_preds).astype(int)))

    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    cohen_kappa = cohen_kappa_score(all_labels, all_preds)
    conf_m = confusion_matrix(all_labels, all_preds)

    print(" Overall Accuracy= " + "{:.4f}".format(acc) +
          " Normalized Accuracy= " + "{:.4f}".format(bacc) +
          " Precision= " + "{:.4f}".format(prec) +
          " Cohen's Kappa= " + "{:.4f}".format(cohen_kappa) +
          " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
          )

    bacc_unknown = 0
    if open_set_class is not None:
        print('Evaluating unknown class', np.bincount(true_map[true_map == open_set_class].ravel()), 
              np.bincount(pred_map[true_map == open_set_class].ravel()))
        bg_mask = true_map != num_classes  # background mask
        true_map_clean = true_map[bg_mask]
        pred_map_clean = pred_map[bg_mask]

        # create a binary mask for the open set class
        y_true_bin = (true_map_clean == open_set_class).astype(int)
        y_pred_bin = (pred_map_clean == open_set_class).astype(int)

        acc_unknown = accuracy_score(y_true_bin, y_pred_bin)
        bacc_unknown  = balanced_accuracy_score(y_true_bin, y_pred_bin)
        prec_unknown  = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        cohen_kappa_unknown = cohen_kappa_score(y_true_bin, y_pred_bin)
        conf_m_unknown = confusion_matrix(y_true_bin, y_pred_bin)

        print(" Unknown Overall Accuracy= " + "{:.4f}".format(acc_unknown) +
            " Unknown Normalized Accuracy= " + "{:.4f}".format(bacc_unknown) +
            " Unknown Precision= " + "{:.4f}".format(prec_unknown) +
            " Unknown Cohen's Kappa= " + "{:.4f}".format(cohen_kappa_unknown) +
            " Unknown Confusion Matrix= " + np.array_str(conf_m_unknown).replace("\n", "")
            )
        
    return bacc, bacc_unknown


def compute_class_weights(labels, num_classes, ignore_index, method="median_freq"):
    """
    Compute class weights for cross-entropy loss.

    method : 'inverse_freq' | 'inverse_sqrt' | 'median_freq' | 'effective_num'
    """
    labels_np = np.asarray(labels).flatten()
    labels_np = labels_np[labels_np != ignore_index]

    # ── count samples per class ───────────────────────────────────────────────
    counts = np.array([(labels_np == c).sum() for c in range(num_classes)], dtype=np.float64)
    total  = counts.sum()
    freq   = counts / total

    # print("Class counts:", {c: int(counts[c]) for c in range(num_classes)})

    if method == "inverse_freq":
        weights = 1.0 / (freq + 1e-6)
    elif method == "inverse_sqrt":
        weights = 1.0 / (np.sqrt(freq) + 1e-6)
    elif method == "median_freq":
        # w_c = median(freq) / freq_c (SegNet paper)
        weights = np.median(freq) / (freq + 1e-6)
    elif method == "effective_num":
        # from "Class-Balanced Loss Based on Effective Number of Samples" CVPR 2019
        beta    = (total - 1.0) / total
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts + 1e-6))
    else:
        raise ValueError(f"Unknown method '{method}'.")

    weights = weights / weights.sum() * num_classes
    # print(f"Class weights ({method}):", {c: round(weights[c], 4) for c in range(num_classes)})

    return torch.tensor(weights, dtype=torch.float32)


def manipulate_itirapina_mask(mask):
    print(mask.shape, np.bincount(mask.ravel()))
    mask[mask == 0] = 13
    mask = mask - 1
    print(mask.shape, np.bincount(mask.ravel()))
    imageio.imwrite("/Users/keillernogueira/Downloads/whole_mask_int_itirapina_v2_background_swapped.png", mask)


def check_itirapina_dataset(path):
    all_files = sorted(os.listdir(path))
    print(all_files)
    d = defaultdict(list)
    for f in all_files:
        _, day, hour = f[:-4].split("_")
        d[day].append(int(hour))
    # sort dict based on key
    d = dict(sorted(d.items()))
    print(d)
    for day, hours in d.items():
        if len(hours) != 13:
            print(day, sorted(hours), len(hours))

if __name__ == '__main__':
    # mask = imageio.v2.imread("/Users/keillernogueira/Downloads/whole_mask_int_itirapina_v2.png")
    # manipulate_itirapina_mask(mask)
    check_itirapina_dataset("/Users/keillernogueira/Downloads/images/1/")
