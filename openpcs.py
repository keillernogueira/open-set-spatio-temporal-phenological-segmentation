import os
import numpy as np

from math import log
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.mixture import GaussianMixture
from scipy.linalg import fractional_matrix_power, pinvh

from utils import *
from networks import GRSL

import torch
from torch.autograd import Variable
import torch.nn.functional as F

quantiles = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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
    # features = np.dot(features, fractional_matrix_power(covariance_matrix, -1/2))  # 1x16 * 16x16
    # print(f"cov eigenvalues: {np.linalg.eigvalsh(covariance_matrix)}")  # any near zero?
    
    wp = fractional_matrix_power(covariance_matrix, -1/2)
    print(f"whitening matrix has nan: {np.isnan(wp).any()}")
    print(f"whitening matrix has inf: {np.isinf(wp).any()}")
    
    features = np.dot(features, wp)
    print(f"whitened features has nan: {np.isnan(features).any()}")
    print(f"whitened features has inf: {np.isinf(features).any()}")
    
    # cov_reg = covariance_matrix + eps * np.eye(covariance_matrix.shape[0])
    # features = np.dot(features, fractional_matrix_power(cov_reg, -1/2))
    # return features, np.eye(cov_reg.shape[0])
    
    return features, np.eye(covariance_matrix.shape[0])


def pred_pixelwise(model_full, feat_np, prds_np, num_classes, method="OpenPCS", threshold=None):
    scores = np.zeros_like(prds_np, dtype=np.float32)
    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            if method == "OpenPCS" or method == "OpenGMM":
                scores[feat_msk] = model_full['generative'][c].score_samples(feat_np[feat_msk, :])
                # s = model_full['generative'][c].score_samples(feat_np[feat_msk, :])
                # n_inf = np.isinf(s).sum()
                # n_nan = np.isnan(s).sum()
                # print(f"Class {c}: {feat_msk.sum()} samples | -inf: {n_inf} | nan: {n_nan} | min: {np.min(s)} | max: {np.max(s)}")
            elif method == "OpenPCS++":
                feats = model_full['generative'][c].transform(feat_np[feat_msk, :])
                feats_pca, cov_matrix = cov_matrix_identity(feats, model_full['cov_matrix'][c])
                scores[feat_msk] = score_loglike(feats_pca, cov_matrix)

    # prds_np[scores < threshold] = num_classes
    # return prds_np, scores
    return scores


def fit_pca_model(feat_np, true_np, prds_np, cl, n_components, method="OpenPCS", limit_samples=False):
    # model = decomposition.PCA(n_components=n_components, random_state=12345)
    if method == "OpenPCS" or method == "OpenPCS++":
        model = decomposition.PCA(n_components=n_components, svd_solver='full', random_state=12345)
    else:
        model = GaussianMixture(n_components=n_components, verbose=2, covariance_type='tied', init_params='random', reg_covar=0.0001, 
                                random_state=12345)

    cl_feat_flat = feat_np[(true_np == cl) & (prds_np == cl), :]

    perm = np.random.permutation(cl_feat_flat.shape[0])
    if limit_samples and perm.shape[0] > 32768:
        cl_feat_flat = cl_feat_flat[perm[:32768], :]

    model.fit(cl_feat_flat)
    # print(f"Explained variance ratios: {model.explained_variance_ratio_}")
    # print(f"Eigenvalues: {model.explained_variance_}")
    # print(f"Any near-zero eigenvalues: {(model.explained_variance_ < 1e-10).any()}")

    # OpenPCS++
    covariance_matrix = None
    if method == "OpenPCS++":
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

    # thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    #               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    scr_thresholds = np.quantile(scores, quantiles).tolist()

    return scr_thresholds


def train_openset(train_loader, net, n_components=64, open_set_method="OpenPCS"):
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
        model, conv_matrix = fit_pca_model(feature_list, label_list, pred_list, c, n_components, method=open_set_method)
        model_list.append(model)
        conv_matrixes.append(conv_matrix)

    scr_thresholds = fit_quantiles(model_list, feature_list, pred_list, train_loader.dataset.num_classes - 1)
    print(scr_thresholds)

    return {'generative': model_list, 'cov_matrix': conv_matrixes, 'thresholds': scr_thresholds}


def test_openset(loader, net, model_full, hidden_class, output_path, open_set_method="OpenPCS"):
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

        scores = pred_pixelwise(model_full, features, preds_numpy, num_classes=loader.dataset.num_classes-1, method=open_set_method)

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
    print('open set all 1', pred_image.shape, np.bincount(pred_image.ravel()))
    pred_image[pred_image == 0] = loader.dataset.num_classes
    for i in range(1, loader.dataset.num_classes):
        pred_image[pred_image == i] = i - 1
    pred_image[loader.dataset.open_set_mask == 4] = 4  # background

    print('open set all', pred_image.shape, np.bincount(pred_image.ravel()), 
          loader.dataset.open_set_mask.shape, np.bincount(loader.dataset.open_set_mask.ravel()),
          score_open_set_norm.shape, np.min(score_open_set_norm), np.max(score_open_set_norm))

    # Saving original predictions.
    imageio.imwrite(os.path.join(output_path, 'prediction_closed_set.png'), 
                    create_lookup_class(loader.dataset.num_classes, hidden_class=hidden_class)[pred_image])

    evaluate_openset(model_full['thresholds'], pred_image, score_open_set_norm, loader.dataset.open_set_mask, 
                     loader.dataset.hidden_class, loader.dataset.num_classes, loader.dataset.open_set_class, output_path)


def evaluate_openset(thresholds, current_preds, openset_score, open_set_mask, hidden_class, 
                     num_classes, open_set_class, output_path, save_images=True):
    evaluate_roc_auc(open_set_mask, openset_score, num_classes, open_set_class, output_path)
    
    print('thresholds', thresholds)
    
    for i, t in enumerate(thresholds):
        open_set_pred = np.copy(current_preds)
        open_set_pred[openset_score < t] = open_set_class  # open set
        open_set_pred[open_set_mask == num_classes] = num_classes  # background
        
        print(f'-------------------------------{quantiles[i]}---------------------------------------------')
        print('openset eval', current_preds.shape, np.bincount(current_preds.ravel()), 
              open_set_mask.shape, np.bincount(open_set_mask.ravel()),
              open_set_pred.shape, np.bincount(open_set_pred.ravel()))
        bacc, bacc_unknown = evaluate_map(open_set_mask, open_set_pred, num_classes, hidden_class, open_set_class=open_set_class)
        print(f'Average Balanced Accuracy: {(3*bacc+bacc_unknown)/4:.4f}')

        if save_images:
            imageio.imwrite(os.path.join(output_path, f'prediction_open_set_thrindex_{quantiles[i]}_thrvalue_{t:.4f}.png'), 
                            create_lookup_class(num_classes, hidden_class)[open_set_pred])


def evaluate_roc_auc(y_true_open, y_score_open, num_classes, open_set_class, output_path):
    all_labels = None
    all_openset_scores = None
    for c in range(num_classes-1):
        if all_labels is None:
            all_labels = y_true_open[y_true_open == c]
            all_openset_scores = y_score_open[y_true_open == c]
        else:
            all_labels = np.concatenate((all_labels, y_true_open[y_true_open == c]))
            all_openset_scores = np.concatenate((all_openset_scores, y_score_open[y_true_open == c]))

    all_labels = np.concatenate((all_labels, y_true_open[y_true_open == open_set_class]))
    all_openset_scores = np.concatenate((all_openset_scores, y_score_open[y_true_open == open_set_class]))
    
    print('ROC AUC eval', all_labels.shape, np.bincount(all_labels.astype(int)), 
          all_openset_scores.shape, np.min(all_openset_scores), np.max(all_openset_scores))
    
    # all_labels[all_labels != open_set_class] = 0  # known classes as negative
    # all_labels[all_labels == open_set_class] = 1  # open set as positive class
    all_labels = np.where(all_labels == open_set_class, 0, 1)

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
    

# H-score - https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600562.pdf
def evaluate_map(y_true, y_pred, num_classes, hidden_class, open_set_class=None):
    # Exclude background and open set class from evaluation
    valid_classes = [c for c in range(num_classes) if c != hidden_class and c != open_set_class]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true.ravel(), y_pred.ravel(), labels=valid_classes)
    
    # Calculate per-class accuracy
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
    
    # Calculate balanced accuracy for known classes
    bacc = np.mean(per_class_acc)
    
    # Calculate balanced accuracy for unknown class (if open_set_class is defined)
    bacc_unknown = None
    if open_set_class is not None:
        unknown_mask = (y_true == open_set_class)
        if np.any(unknown_mask):
            bacc_unknown = np.mean(y_pred[unknown_mask] == open_set_class)

    print(f'Balanced Accuracy (Known Classes): {bacc:.4f}')
    if bacc_unknown is not None:
        print(f'Balanced Accuracy (Unknown Class): {bacc_unknown:.4f}')
    
    return bacc, bacc_unknown
