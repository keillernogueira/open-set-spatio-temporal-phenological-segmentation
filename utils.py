import os
import argparse
import imageio
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, cohen_kappa_score

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
    print('lookup class', lookup_class)
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
        acc_unknown = accuracy_score(true_map[true_map == open_set_class], pred_map[true_map == open_set_class])
        bacc_unknown = balanced_accuracy_score(true_map[true_map == open_set_class], pred_map[true_map == open_set_class])
        prec_unknown = precision_score(true_map[true_map == open_set_class], pred_map[true_map == open_set_class], average='weighted')
        cohen_kappa_unknown = cohen_kappa_score(true_map[true_map == open_set_class], pred_map[true_map == open_set_class])
        conf_m_unknown = confusion_matrix(true_map[true_map == open_set_class], pred_map[true_map == open_set_class])

        print(" Unknown Overall Accuracy= " + "{:.4f}".format(acc_unknown) +
            " Unknown Normalized Accuracy= " + "{:.4f}".format(bacc_unknown) +
            " Unknown Precision= " + "{:.4f}".format(prec_unknown) +
            " Unknown Cohen's Kappa= " + "{:.4f}".format(cohen_kappa_unknown) +
            " Unknown Confusion Matrix= " + np.array_str(conf_m_unknown).replace("\n", "")
            )
        
    return bacc, bacc_unknown

