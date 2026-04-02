import os
import numpy as np

from math import log
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score
from scipy.linalg import fractional_matrix_power, pinvh

from utils import *
from networks import GRSL

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def fast_logdet(matrix):
    """Compute log(det(A)) for A symmetric.
    Equivalent to : np.log(nl.det(A)) but more robust.
    It returns -Inf if det(A) is non positive or is not defined.
    Parameters
    ----------
    matrix : array-like
        The matrix.
    """
    sign, ld = np.linalg.slogdet(matrix)
    if not sign > 0:
        return -np.inf
    return ld


def score_loglike(data, covariance_matrix):
    """Return the log-likelihood of each sample.
    See. "Pattern Recognition and Machine Learning"
    by C. Bishop, 12.2.1 p. 574
    or http://www.miketipping.com/papers/met-mppca.pdf
    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data.
    covariance_matrix : covariance matrix
    Returns
    -------
    ll : ndarray of shape (n_samples,)
        Log-likelihood of each sample under the current model.
    """
    inv_covariance_matrix = pinvh(covariance_matrix, check_finite=False)
    n_features = data.shape[1]
    log_like = -.5 * (data * (np.dot(data, inv_covariance_matrix))).sum(axis=1)
    log_like -= .5 * (n_features * log(2. * np.pi) - fast_logdet(inv_covariance_matrix))
    return log_like


def cov_matrix_identity(features, covariance_matrix):
    features = np.dot(features, fractional_matrix_power(covariance_matrix, -1/2))  # 1x16 * 16x16
    return features, np.eye(covariance_matrix.shape[0])


def pred_pixelwise(model_full, feat_np, prds_np, num_classes, threshold=None, open_pcs_plus=False):
    scores = np.zeros_like(prds_np, dtype=np.float32)
    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            # print('feat class', c)
            if open_pcs_plus is False:
                scores[feat_msk] = model_full['generative'][c].score_samples(feat_np[feat_msk, :])
            else:
                feats = model_full['generative'][c].transform(feat_np[feat_msk, :])
                feats_pca, cov_matrix = cov_matrix_identity(feats, model_full['cov_matrix'][c])
                scores[feat_msk] = score_loglike(feats_pca, cov_matrix)

    # prds_np[scores < threshold] = num_classes
    # return prds_np, scores
    return scores


def fit_pca_model(feat_np, true_np, prds_np, cl, n_components):
    model = decomposition.PCA(n_components=n_components, random_state=12345)

    cl_feat_flat = feat_np[(true_np == cl) & (prds_np == cl), :]

    perm = np.random.permutation(cl_feat_flat.shape[0])
    if perm.shape[0] > 32768:
        cl_feat_flat = cl_feat_flat[perm[:32768], :]

    model.fit(cl_feat_flat)

    # OpenPCS++
    x_pca_train = model.transform(cl_feat_flat)
    covariance_matrix = np.cov(x_pca_train, rowvar=False)

    return model, covariance_matrix


def fit_quantiles(model_list, feat_np, prds_np, num_classes):
    # Acquiring scores for training set sample.
    scores = np.zeros_like(prds_np, dtype=np.float32)

    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            # print('feat class', c)
            scores[feat_msk] = model_list[c].score_samples(feat_np[feat_msk, :])
            # print(c, feat_np[feat_msk, :].shape, np.isinf(scores).any(), np.min(scores), np.max(scores))

    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    scr_thresholds = np.quantile(scores, thresholds).tolist()

    return scr_thresholds


def train_openset(train_loader, net, n_components=64):
    # Setting network for training mode.
    net.eval()

    # Iterating over batches.
    feature_list = None
    pred_list = None
    label_list = None
    for i, data in enumerate(train_loader):
        # Obtaining data and labels
        inputs, labels = data[0], data[1]

        # Casting tensors to cuda.
        inputs_c, labels_c = inputs.cuda(), labels.cuda()
        # Casting to cuda variables.
        inps = Variable(inputs_c).cuda()
        labs = Variable(labels_c).cuda()

        # Forwarding.
        outs, fc1, fc2 = net(inps.permute(1, 0, 2, 3, 4))  # permute to change the branches with the batch size
        # Computing loss.
        soft_outs = F.softmax(outs, dim=1)
        # Obtaining predictions.
        prds = soft_outs.data.max(1)[1]

        # print('1', prds.shape, labs.shape, outs.shape, fc1.shape, fc2.shape)

        if feature_list is None:
            feature_list = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
            pred_list = prds.detach().cpu().numpy()
            label_list = labs.detach().cpu().numpy()
        else:
            features = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
            feature_list = np.concatenate((feature_list, features))
            pred_list = np.concatenate((pred_list, prds.detach().cpu().numpy()))
            label_list = np.concatenate((label_list, labs.detach().cpu().numpy()))

    if type(net) is not GRSL:
        n, c, h, w = feature_list.shape
        feature_list = np.reshape(np.transpose(feature_list, (0, 2, 3, 1)), (n * h * w, c))
        pred_list = pred_list.ravel()
        label_list = label_list.ravel()

    feature_list = normalize(np.asarray(feature_list), norm='l2', axis=1, copy=False)
    pred_list = np.asarray(pred_list)
    label_list = np.asarray(label_list)
    # print('2', feature_list.shape, pred_list.shape, label_list.shape)
    # print('3', np.bincount(label_list), np.bincount(pred_list))

    model_list = []
    conv_matrixes = []
    for c in range(train_loader.dataset.num_classes - 1):
        print('Fitting model for class %d...' % c)
        # Computing PCA models from features.
        model, conv_matrix = fit_pca_model(feature_list, label_list, pred_list, c, n_components)
        model_list.append(model)
        conv_matrixes.append(conv_matrix)

    scr_thresholds = fit_quantiles(model_list, feature_list, pred_list, train_loader.dataset.num_classes - 1)
    print(scr_thresholds)

    return {'generative': model_list, 'cov_matrix': conv_matrixes, 'thresholds': scr_thresholds}


def test_openset(loader, net, model_full, hidden_class, output_path):
    h, w = loader.dataset.mask.shape
    prob_original = np.zeros((4, h, w), dtype=np.float32)
    score_open_set = np.zeros((h, w), dtype=np.float32)
    occur_im = np.zeros((4, h, w), dtype=int)

    # Setting network for training mode.
    net.eval()

    # Iterating over batches.
    for i, data in enumerate(loader):
        # Obtaining data and labels
        inputs, labels, pos = data[0], data[1], data[2]

        # Casting tensors to cuda.
        inputs_c = inputs.cuda()

        # Casting to cuda variables.
        inps = Variable(inputs_c).cuda()

        # Forwarding.
        outs, fc1, fc2 = net(inps.permute(1, 0, 2, 3, 4))  # permute to change the branches with the batch size
        # print(outs.shape, fc1.shape, fc2.shape)

        # Obtaining predictions.
        soft_outs = F.softmax(outs, dim=1)
        prds = soft_outs.data.max(1)[1]
        preds_numpy = prds.detach().cpu().numpy()

        # Concatenating features
        features = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
        if type(net) is not GRSL:
            features = np.transpose(features, (0, 2, 3, 1))
            features = np.reshape(features, (-1, features.shape[-1]))
            preds_numpy = preds_numpy.ravel()
        features = normalize(np.asarray(features), norm='l2', axis=1, copy=False)
        # print('1', soft_outs.shape, preds_numpy.shape, np.bincount(preds_numpy), features.shape)

        scores = pred_pixelwise(model_full, features, preds_numpy, loader.dataset.num_classes-1)

        if type(net) is GRSL:
            for j, p in enumerate(pos):
                prob_original[p[0], p[1]] = preds_numpy[j]
                # pred_post[p[0], p[1]] = prds_post[j]
                score_open_set[p[0], p[1]] = scores[j]
        else:
            # recreating the entire image from patches
            scores = scores.reshape((-1, loader.dataset.patch_size, loader.dataset.patch_size))
            for j, (cur_x, cur_y) in enumerate(pos):
                prob_original[1:4, cur_x:cur_x + loader.dataset.patch_size,
                              cur_y:cur_y + loader.dataset.patch_size] += soft_outs[j, :, :, :].detach().cpu().numpy()
                score_open_set[cur_x:cur_x + loader.dataset.patch_size,
                               cur_y:cur_y + loader.dataset.patch_size] += scores[j, :, :]
                occur_im[:, cur_x:cur_x + loader.dataset.patch_size, cur_y:cur_y + loader.dataset.patch_size] += 1

    # calculating the average
    occur_im[np.where(occur_im == 0)] = 1
    pred_image = np.argmax(prob_original / occur_im.astype(float), axis=0)
    score_open_set_norm = score_open_set / occur_im.astype(float)[0, :, :]

    # moving background to correct position and rearranging class so hidden class is empty
    pred_image[pred_image == 0] = loader.dataset.num_classes
    for i in range(1, hidden_class+1):
        pred_image[pred_image == i] = i - 1
    pred_image[loader.dataset.open_set_mask == 4] = 4  # background

    print('3', pred_image.shape, np.bincount(pred_image.ravel()), 
          loader.dataset.open_set_mask.shape, np.bincount(loader.dataset.open_set_mask.ravel()),
          score_open_set_norm.shape, np.min(score_open_set_norm), np.max(score_open_set_norm))
    # Saving original predictions.
    imageio.imwrite(os.path.join(output_path, 'prediction_closed_set.png'), lookup_class[pred_image])

    evaluate_openset(model_full['thresholds'], pred_image, score_open_set_norm, loader.dataset.open_set_mask, 
                     loader.dataset.hidden_class, loader.dataset.num_classes, loader.dataset.open_set_class, output_path)


def evaluate_openset(thresholds, current_preds, openset_score, open_set_mask, hidden_class, 
                     num_classes, open_set_class, output_path, save_images=False):
    evaluate_roc_auc(open_set_mask, openset_score, num_classes, hidden_class, open_set_class, output_path)
    
    print('thresholds', thresholds)
    quantiles = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    for i, t in enumerate(thresholds):
        open_set_pred = np.copy(current_preds)
        open_set_pred[openset_score < t] = open_set_class  # open set
        open_set_pred[open_set_mask == num_classes] = num_classes  # background
        
        print(f'-------------------------------{t}---------------------------------------------')
        print('openset eval', open_set_mask.shape, np.bincount(open_set_mask.ravel()), 
              open_set_pred.shape, np.bincount(open_set_pred.ravel()))
        bacc, bacc_unknown =evaluate_map(open_set_mask, open_set_pred, hidden_class, num_classes, open_set_class=open_set_class)
        print(f'Average Balanced Accuracy: {(bacc+bacc_unknown)/2:.4f}')

        if save_images:
            imageio.imwrite(os.path.join(output_path, f'prediction_open_set_thrindex_{quantiles[i]}_thrvalue_{t:.4f}.png'), 
                            lookup_class[open_set_pred])


def evaluate_roc_auc(y_true_open, y_score_open, num_classes, hidden_class, open_set_class, output_path):
    all_labels = None
    all_openset_scores = None
    for c in range(num_classes):
        if c == hidden_class:
            continue
        if all_labels is None:
            all_labels = y_true_open[y_true_open == c]
            all_openset_scores = y_score_open[y_true_open == c]
        else:
            all_labels = np.concatenate((all_labels, y_true_open[y_true_open == c]))
            all_openset_scores = np.concatenate((all_openset_scores, y_score_open[y_true_open == c]))

    all_labels = np.concatenate((all_labels, y_true_open[y_true_open == open_set_class]))
    all_openset_scores = np.concatenate((all_openset_scores, y_score_open[y_true_open == open_set_class]))
    
    # all_labels[all_labels != open_set_class] = 0  # known classes as negative
    # all_labels[all_labels == open_set_class] = 1  # open set as positive class
    all_labels = np.where(all_labels == open_set_class, 0, 1)  # open set as positive class, known classes as negative

    fpr_vals, tpr_vals, roc_thresholds = roc_curve(all_labels, all_openset_scores, pos_label=1)
    auc_val = auc(fpr_vals, tpr_vals)

    print(f"Open-set ROC AUC: {auc_val:.4f}")
    print(f"ROC thresholds: {roc_thresholds}")

    #plot the curve
    plt.figure()
    plt.plot(fpr_vals, tpr_vals, color='blue', lw=2, label=f'ROC curve (area = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_path, 'open_set_roc_curve.png'))
    plt.close()
    
