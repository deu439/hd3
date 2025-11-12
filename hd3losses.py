import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.special as ss

from models.hd3_ops import *


class LossCalculator(object):

    def __init__(self, task):
        assert task in ['flow', 'stereo']
        self.task = task
        self.dim = 1 if task == 'stereo' else 2

    def __call__(self, ms_prob, ms_pred, gt, corr_range, ds=6):
        B, C, H, W = gt.size()
        lv = len(ms_prob)
        criterion = nn.KLDivLoss(reduction='batchmean').cuda()
        losses = {}
        kld_loss = 0
        for l in range(lv):
            scaled_gt, valid_mask = downsample_flow(gt, 1 / 2**(ds - l))
            if self.task == 'stereo':
                scaled_gt = scaled_gt[:, 0, :, :].unsqueeze(1)
            if l > 0:
                scaled_gt = scaled_gt - F.interpolate(
                    ms_pred[l - 1],
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True)
            scaled_gt = scaled_gt / 2**(ds - l)
            gt_dist = vector2density(scaled_gt, corr_range[l],
                                     self.dim) * valid_mask
            kld_loss += 4**(ds - l) / (H * W) * criterion(
                F.log_softmax(ms_prob[l], dim=1), gt_dist.detach())

        losses['total'] = kld_loss
        for loss_type, loss_value in losses.items():
            losses[loss_type] = loss_value.reshape(1)
        return losses


def EndPointError(output, gt):
    # output: [B, 1/2, H, W], stereo or flow prediction
    # gt: [B, C, H, W], 2D ground-truth annotation which may contain a mask
    # NOTE: To benchmark the result, please ensure the ground-truth keeps
    # its ORIGINAL RESOLUTION.
    if output.size(1) == 1:  # stereo
        output = disp2flow(output)
    output = resize_dense_vector(output, gt.size(2), gt.size(3))
    error = torch.norm(output - gt[:, :2, :, :], 2, 1, keepdim=False) # [B, H, W]
    magnitude = torch.norm(gt[:, :2, :, :], 2, 1, keepdim=False)    # [B, H, W]

    if gt.size(1) == 4: # KITTI data with occ/noc masks
        mask_occ = (gt[:, 2, :, :] > 0).float() # [B, H, W]
        mask_noc = (gt[:, 3, :, :] > 0).float() # [B, H, W]
        epe_occ = (error * mask_occ).sum() / mask_occ.sum()
        epe_noc = (error * mask_noc).sum() / mask_noc.sum()
        cond1 = error > 3.0
        cond2 = (error / torch.max(magnitude, torch.tensor(1e-10, device=magnitude.device))) > 0.05
        bad_pixels = (cond1 & cond2).float()
        error_rate = 100 * (bad_pixels * mask_occ).sum() / mask_occ.sum()

        return epe_occ.reshape(1), epe_noc.reshape(1), error_rate.reshape(1)
    else:
        return error.mean()


def sp_plot(error, entropy, gt_mask, n=25, alpha=100.0, eps=1e-1):
    def sp_mask(thr, entropy, gt_mask):
        mask = ss.expit(alpha * (thr[:, None, None] - entropy[None, :, :]))
        frac = np.sum((1.0 - mask)*gt_mask[None], axis=(1, 2)) / np.sum(gt_mask)[None]
        return mask*gt_mask[None], frac

    # Find the primary interval for soft thresholding
    greatest = np.max(entropy) + eps    # Avoid zero-sized interval
    least = np.min(entropy) - eps
    _, frac = sp_mask(np.array([least]), entropy, gt_mask)
    while abs(frac.item() - 1.0) > eps:
        least -= 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([least]), entropy, gt_mask)

    _, frac = sp_mask(np.array([greatest]), entropy, gt_mask)
    while abs(frac.item() - 0.0) > eps:
        greatest += 1e-3*(greatest - least)
        _, frac = sp_mask(np.array([greatest]), entropy, gt_mask)

    # Approximate uniform grid
    grid_entr = np.linspace(greatest, least, n)
    grid_frac = np.linspace(0, 1, n)
    mask, frac = sp_mask(grid_entr, entropy, gt_mask)
    for i in range(10):
        #print("res: ", np.max(np.abs(frac - grid_frac)))
        if np.max(np.abs(frac - grid_frac)) <= eps:
            break
        grid_entr = np.interp(grid_frac, frac, grid_entr)
        mask, frac = sp_mask(grid_entr, entropy, gt_mask)

    # Check whether the grid is approximately uniform
    if np.max(np.abs(frac - grid_frac)) > eps:
        print("Warning! sp_plot did not converge!")
        #raise RuntimeError("sp_plot did not converge!")

    # Calculate the sparsification plot
    splot = np.sum(error[None, :, :] * mask, axis=(1,2)) / np.sum(mask, axis=(1,2))

    # Resample on uniform grid
    splot = np.interp(grid_frac, frac, splot)

    return splot


def evaluate_uncertainty(gt_flows, pred_flows, pred_entropies, sp_samples=100):
    auc, oracle_auc = 0, 0
    splots, oracle_splots = [], []
    batch_size = len(gt_flows)
    for gt_flow, pred_flow, pred_entropy, i in zip(gt_flows, pred_flows, pred_entropies, range(batch_size)):
        # Calculate sparsification plots
        epe_map = np.sqrt(np.sum(np.square(pred_flow[:, :, :2] - gt_flow[:, :, :2]), axis=2))
        if gt_flow.shape[2] == 4:    # KITTY dataset includes a mask in the third dimension
            mask = (gt_flow[:, :, 2] > 0).astype(np.float32)
        else:
            mask = np.ones_like(epe_map)
        entropy_map = np.sum(pred_entropy[:, :, :2], axis=2)
        splot = sp_plot(epe_map, entropy_map, mask, n=sp_samples)
        oracle_splot = sp_plot(epe_map, epe_map, mask, n=sp_samples)     # Oracle

        # Collect the sparsification plots and oracle sparsification plots
        splots += [splot]
        oracle_splots += [oracle_splot]

        # Cummulate AUC
        frac = np.linspace(0, 1, sp_samples)
        auc += np.trapz(splot / splot[0], x=frac)
        oracle_auc += np.trapz(oracle_splot / oracle_splot[0], x=frac)

    return [auc / batch_size, (auc - oracle_auc) / batch_size], splots, oracle_splots


def evaluate_auc(prob, vec, gt):
    dim = vec.size(1)
    device = prob.device
    prob = prob_gather(prob, normalize=True, dim=dim)

    # Resize to match the ground-truth size
    vec = resize_dense_vector(vec, gt.size(2), gt.size(3))
    prob = F.interpolate(prob, (gt.size(2), gt.size(3)), mode='nearest')
    prob = prob.repeat(1, 2, 1, 1)

    # To numpy
    gt = gt.cpu().numpy().transpose(0, 2, 3, 1)
    vec = vec.cpu().numpy().transpose(0, 2, 3, 1)
    prob = prob.cpu().numpy().transpose(0, 2, 3, 1)
    (auc, rel_auc), _, _ = evaluate_uncertainty(gt, vec, 1-prob)
    return torch.tensor([auc], device=device), torch.tensor([rel_auc], device=device)
