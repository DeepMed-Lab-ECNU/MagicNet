import os.path
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from scipy.ndimage import zoom
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def validation_all_case(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16):
    # ex: base_dir: /data/btcv_h5
    loader = tqdm(image_list)
    # total_dice = np.zeros(num_classes - 1)
    # total_dice = []
    total_metric = []
    for case_idx in loader:
        # img_path = os.path.join(base_dir + '/img', 'img' + case_idx + '.nii.gz')
        # lab_path = os.path.join(base_dir + '/lab', 'lab' + case_idx + '.nii.gz')
        # image, gt_mask = nib.load(img_path).get_fdata(), nib.load(lab_path).get_fdata()
        image_path = base_dir + '/{}.h5'.format(case_idx)
        h5f = h5py.File(image_path, 'r')
        image, gt_mask = h5f['image'][:], h5f['label'][:]
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction) == 0:
            # dice = np.zeros(num_classes - 1)
            case_metric = np.zeros((4, num_classes - 1))
        else:
            # dice = cal_dice(prediction, gt_mask, num_classes)
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        # total_dice += dice
        # total_dice.append(np.expand_dims(dice, axis=0))
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8
    all_metric = np.concatenate(total_metric, axis=0)
    # all_dice = np.concatenate(total_dice, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def validation_all_case_mact(model, num_classes, base_dir, image_list, patch_size=(96, 96, 96), stride_xy=16, stride_z=16):
    # ex: base_dir: /data/MACT
    image_list = ['{:0>4}'.format(item + 1) for item in image_list]
    loader = tqdm(image_list)
    # total_dice = np.zeros(num_classes - 1)
    # total_dice = []
    total_metric = []
    for case_idx in loader:
        # img_path = os.path.join(base_dir + '/img', 'img' + case_idx + '.nii.gz')
        # lab_path = os.path.join(base_dir + '/lab', 'lab' + case_idx + '.nii.gz')
        # image, gt_mask = nib.load(img_path).get_fdata(), nib.load(lab_path).get_fdata()
        image_path = base_dir + '/{}.h5'.format(case_idx)
        h5f = h5py.File(image_path, 'r')
        image, gt_mask = h5f['image'][:], h5f['label'][:]
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction) == 0:
            # dice = np.zeros(num_classes - 1)
            case_metric = np.zeros((4, num_classes - 1))
        else:
            # dice = cal_dice(prediction, gt_mask, num_classes)
            case_metric = np.zeros((4, num_classes - 1))
            for i in range(1, num_classes):
                case_metric[:, i - 1] = cal_metric(prediction == i, gt_mask == i)
        # total_dice += dice
        # total_dice.append(np.expand_dims(dice, axis=0))
        total_metric.append(np.expand_dims(case_metric, axis=0))

    # all_metric:22x4x8
    all_metric = np.concatenate(total_metric, axis=0)
    # all_dice = np.concatenate(total_dice, axis=0)
    avg_dice, std_dice = np.mean(all_metric, axis=0)[0], np.std(all_metric, axis=0)[0]
    return avg_dice, std_dice, all_metric


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    # score_map : CxDxHxW, cnt: DxHxW
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
    # score map: CxDxHxW, label_map: CxDxHxW -> DxHxW
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map


def cal_metric(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        sf = compute_surface_distances(gt, pred, spacing_mm=(1., 1., 1.))
        nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
        return np.array([dice, hd95, nsd, asd])
    else:
        return np.zeros(4)


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        total_dice[i - 1] += metric.binary.dc(prediction_tmp, label_tmp)

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def dice_ratio(prediction, label, num=2):
    """
    dice ratio
    :param masks:
    :param labels:
    :return:
    """
    masks = prediction.cpu().data.numpy()
    labels = label.cpu().data.numpy()
    bs = masks.shape[0]
    total_dice = np.zeros(num - 1)
    for i in range(bs):
        case_dice_tmp = cal_dice(masks[i, :], labels[i, :], num)
        total_dice += case_dice_tmp

    return total_dice / bs


