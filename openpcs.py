import sys
import os
import joblib
import numpy as np
import imageio

from utils import cedro

from sklearn import decomposition

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def pred_pixelwise(model_full, feat_np, prds_np, num_classes, threshold):
    scores = np.zeros_like(prds_np, dtype=np.float)
    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            scores[feat_msk] = model_full['generative'][c].score_samples(feat_np[feat_msk, :])

    prds_np[scores < threshold] = num_classes
    return prds_np, scores


def fit_pca_model(feat_np, true_np, prds_np, cl, n_components):
    model = decomposition.PCA(n_components=n_components, random_state=12345)

    cl_feat_flat = feat_np[(true_np == cl) & (prds_np == cl), :]

    perm = np.random.permutation(cl_feat_flat.shape[0])
    if perm.shape[0] > 32768:
        cl_feat_flat = cl_feat_flat[perm[:32768], :]

    model.fit(cl_feat_flat)
    return model


def fit_quantiles(model_list, feat_np, prds_np, num_classes):
    # Acquiring scores for training set sample.
    scores = np.zeros_like(prds_np, dtype=np.float)

    for c in range(num_classes):
        feat_msk = (prds_np == c)
        if np.any(feat_msk):
            scores[feat_msk] = model_list[c].score_samples(feat_np[feat_msk, :])

    thresholds = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    scr_thresholds = np.quantile(scores, thresholds).tolist()

    return scr_thresholds


def train_openset(train_loader, net, num_classes, output_path, n_components=16):
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

        # print(prds.shape, labs.shape, outs.shape, fc1.shape, fc2.shape)

        if feature_list is None:
            feature_list = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
            pred_list = prds.detach().cpu().numpy()
            label_list = labs.detach().cpu().numpy()
        else:
            features = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
            feature_list = np.concatenate((feature_list, features))
            pred_list = np.concatenate((pred_list, prds.detach().cpu().numpy()))
            label_list = np.concatenate((label_list, labs.detach().cpu().numpy()))

    feature_list = np.asarray(feature_list)
    pred_list = np.asarray(pred_list)
    label_list = np.asarray(label_list)
    print(feature_list.shape, pred_list.shape, label_list.shape)

    print(np.isnan(feature_list).any(), np.isnan(pred_list).any(), np.isnan(label_list).any())

    model_list = []
    for c in range(num_classes):
        print('Fitting model for class %d...' % c)
        # Computing PCA models from features.
        model = fit_pca_model(feature_list, label_list, pred_list, c, n_components)
        model_list.append(model)

    scr_thresholds = fit_quantiles(model_list, feature_list, pred_list, num_classes)

    model_full = {'generative': model_list, 'thresholds': scr_thresholds}

    # Saving model on disk.
    sys.stdout.flush()
    joblib.dump(model_full, os.path.join(output_path, 'model_pca.pkl'))
    return model_full


def test_openset(loader, net, model_full, dataset, output_path):
    _, h, w, c = dataset.images.shape
    pred_image = np.empty((h, w, c), dtype=np.uint8)

    # Setting network for training mode.
    net.eval()

    # Iterating over batches.
    for i, data in enumerate(loader):
        # Obtaining data and labels
        inputs, labels, pos = data[0], data[1], data[2]

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
        # print(prds.shape, labs.shape, outs.shape, fc1.shape, fc2.shape)

        feature_list = torch.cat([outs.squeeze(), fc1.squeeze(), fc2.squeeze()], 1).detach().cpu().numpy()
        pred_list = prds.detach().cpu().numpy()
        labs = labs.detach().cpu().numpy()
        print(labs.shape, np.min(labs), np.max(labs), pred_list.shape, np.min(pred_list), np.max(pred_list))

        # print(model_full['thresholds'], model_full['thresholds'][-6])
        prds_post, scores = pred_pixelwise(model_full, feature_list, pred_list, dataset.num_classes,
                                           model_full['thresholds'][-6])
        print(pos.shape, prds_post.shape, np.min(prds_post), np.max(prds_post))

        for i, p in enumerate(pos):
            pred_image[p[0], p[1], :] = cedro[prds_post[i]]

    # Saving predictions.
    imageio.imwrite(os.path.join(output_path, 'prediction.png'), pred_image)
