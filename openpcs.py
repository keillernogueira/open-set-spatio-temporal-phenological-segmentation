import os
import numpy as np

from math import log
from sklearn import decomposition
from sklearn.preprocessing import normalize
from scipy.linalg import fractional_matrix_power, pinvh

from utils import convert_pred_image_to_rgb
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


def pred_pixelwise(model_full, feat_np, prds_np, num_classes, threshold, open_pcs_plus=False):
    scores = np.zeros_like(prds_np, dtype=np.float16)
    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            if open_pcs_plus is False:
                scores[feat_msk] = model_full['generative'][c].score_samples(feat_np[feat_msk, :])
            else:
                feats = model_full['generative'][c].transform(feat_np[feat_msk, :])
                feats_pca, cov_matrix = cov_matrix_identity(feats, model_full['cov_matrix'][c])
                scores[feat_msk] = score_loglike(feats_pca, cov_matrix)

    prds_np[scores < threshold] = num_classes
    return prds_np, scores


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
    scores = np.zeros_like(prds_np, dtype=np.float)

    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
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
        inputs_c.squeeze_(0)
        labels_c.squeeze_(0)

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

    print(feature_list.shape, pred_list.shape, label_list.shape)
    # print(np.isnan(feature_list).any(), np.isnan(pred_list).any(), np.isnan(label_list).any())
    # print(np.min(feature_list), np.max(feature_list))

    model_list = []
    conv_matrixes = []
    for c in range(train_loader.dataset.num_classes):
        print('Fitting model for class %d...' % c)
        # Computing PCA models from features.
        model, conv_matrix = fit_pca_model(feature_list, label_list, pred_list, c, n_components)
        model_list.append(model)
        conv_matrixes.append(conv_matrix)

    scr_thresholds = fit_quantiles(model_list, feature_list, pred_list, train_loader.dataset.num_classes)
    print(scr_thresholds)

    return {'generative': model_list, 'cov_matrix': conv_matrixes, 'thresholds': scr_thresholds}


def test_openset(loader, net, model_full, output_path):
    h, w = loader.dataset.mask.shape

    pred_original = np.full((h, w), fill_value=-1, dtype=np.int)
    # pred_post = np.full((h, w), fill_value=-1, dtype=np.int)
    score_pca = np.full((h, w), fill_value=-1, dtype=np.float16)
    count = 0

    # Setting network for training mode.
    net.eval()

    # Iterating over batches.
    for i, data in enumerate(loader):
        # Obtaining data and labels
        inputs, labels, pos = data[0], data[1], data[2]

        # Casting tensors to cuda.
        inputs_c = inputs.cuda()  # , labels.cuda()
        inputs_c.squeeze_(0)
        # labels_c.squeeze_(0)

        # Casting to cuda variables.
        inps = Variable(inputs_c).cuda()
        # labs = Variable(labels_c).cuda()

        # Forwarding.
        outs, fc1, fc2 = net(inps.permute(1, 0, 2, 3, 4))  # permute to change the branches with the batch size
        # print(outs.shape, fc1.shape, fc2.shape)

        # Obtaining predictions.
        soft_outs = F.softmax(outs, dim=1)
        prds = soft_outs.data.max(1)[1]
        preds_numpy = prds.detach().cpu().numpy()

        print('1', preds_numpy.shape, np.min(preds_numpy), np.max(preds_numpy))

        # Concatenating features
        features = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
        if type(net) is not GRSL:
            features = np.transpose(features, (0, 2, 3, 1))
            features = np.reshape(features, (-1, features.shape[-1]))
            preds_numpy = preds_numpy.ravel()
        features = normalize(np.asarray(features), norm='l2', axis=1, copy=False)

        prds_post, scores = pred_pixelwise(model_full, features, preds_numpy, loader.dataset.num_classes,
                                           model_full['thresholds'][15])
        # print('3', type(scores), type(scores[0]), scores.shape, np.min(scores), np.max(scores),
        #       type(prds_post), type(prds_post[0]), prds_post.shape, np.min(prds_post), np.max(prds_post),
        #       preds_numpy.shape, np.min(preds_numpy), np.max(preds_numpy))

        if type(net) is GRSL:
            for j, p in enumerate(pos):
                pred_original[p[0], p[1]] = preds_numpy[j]
                # pred_post[p[0], p[1]] = prds_post[j]
                score_pca[p[0], p[1]] = scores[j]
        else:
            preds_numpy = prds.detach().cpu().numpy()
            # prds_post = np.reshape(prds_post, (-1, loader.dataset.patch_size, loader.dataset.patch_size))
            scores = np.reshape(scores, (-1, loader.dataset.patch_size, loader.dataset.patch_size))

            print('3', type(scores), type(scores[0]), scores.shape, np.min(scores), np.max(scores),
                  type(prds_post), type(prds_post[0]), prds_post.shape, np.min(prds_post), np.max(prds_post),
                  preds_numpy.shape, np.min(preds_numpy), np.max(preds_numpy))

            for j, p in enumerate(pos):
                for k in range(loader.dataset.patch_size):
                    for m in range(loader.dataset.patch_size):
                        count += 1
                        pred_original[p[0] + k, p[1] + m] = preds_numpy[j, k, m]
                        if score_pca[p[0] + k, p[1] + m] != 1:
                            if score_pca[p[0] + k, p[1] + m] < scores[j, k, m]:
                                # pred_post[p[0] + k, p[1] + m] = prds_post[j, k, m]
                                score_pca[p[0] + k, p[1] + m] = scores[j, k, m]
                        else:
                            # pred_post[p[0] + k, p[1] + m] = prds_post[j, k, m]
                            score_pca[p[0] + k, p[1] + m] = scores[j, k, m]

    print('count', count, pred_original.shape, np.min(pred_original), np.max(pred_original))
    # Saving predictions.
    np.save(os.path.join(output_path, 'prediction_orig.npy'), pred_original)
    # np.save(os.path.join(output_path, 'prediction_post_tpr_15.npy'), pred_post)
    np.save(os.path.join(output_path, 'scores.npy'), score_pca)


def evaluate_tpr(model_full, output_path):
    score_pca = np.load(os.path.join(output_path, 'scores.npy'))
    pred_original = np.load(os.path.join(output_path, 'prediction_orig.npy'))

    thresholds = model_full['thresholds']
    print('thresholds', thresholds)

    quantiles = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    for i, t in enumerate(thresholds):
        preds = np.copy(pred_original)
        unique, counts = np.unique(preds, return_counts=True)
        print('1', dict(zip(unique, counts)), preds.shape, np.min(preds), np.max(preds))

        preds[score_pca < t] = 4
        unique, counts = np.unique(preds, return_counts=True)
        print('2', dict(zip(unique, counts)), preds.shape, np.min(preds), np.max(preds))
        convert_pred_image_to_rgb(preds, os.path.join(output_path, 'prediction_post_tpr_' + str(quantiles[i]) + '.png'))
