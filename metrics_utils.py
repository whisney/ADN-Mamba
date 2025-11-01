import numpy as np
from typing import Optional
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.util.arraycrop import crop

def compute_mae(pred, label, mask=None):
    mae = np.abs(pred - label)
    if mask is not None:
        mae = np.mean(mae[mask == 1])
    else:
        mae = np.mean(mae)
    return mae

def ssim(gt, pred, mask, dynamic_range) -> float:
    """
    Compute Structural Similarity Index Metric (SSIM)

    Parameters
    ----------
    gt : np.ndarray
        Ground truth
    pred : np.ndarray
        Prediction
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).

    Returns
    -------
    ssim : float
        structural similarity index metric.

    """
    if mask is not None:
        # binarize mask
        mask = np.where(mask > 0, 1., 0.)

        # Mask gt and pred
        gt = np.where(mask == 0, min(dynamic_range), gt)
        pred = np.where(mask == 0, min(dynamic_range), pred)

    # Make values non-negative
    if min(dynamic_range) < 0:
        gt = gt - min(dynamic_range)
        pred = pred - min(dynamic_range)

    # Set dynamic range for ssim calculation and calculate ssim_map per pixel
    dynamic_range = dynamic_range[1] - dynamic_range[0]
    ssim_value_full, ssim_map = structural_similarity(gt, pred, data_range=dynamic_range, full=True)

    if mask is not None:
        # Follow skimage implementation of calculating the mean value:
        # crop(ssim_map, pad).mean(dtype=np.float64), with pad=3 by default.
        pad = 3
        ssim_value_masked = (crop(ssim_map, pad)[crop(mask, pad).astype(bool)]).mean(dtype=np.float64)
        return ssim_value_masked
    else:
        return ssim_value_full

def psnr(gt, pred, mask, dynamic_range) -> float:
    """
    Compute Peak Signal to Noise Ratio metric (PSNR)

    Parameters
    ----------
    gt : np.ndarray
        Ground truth
    pred : np.ndarray
        Prediction
    mask : np.ndarray, optional
        Mask for voxels to include. The default is None (including all voxels).
    use_population_range : bool, optional
        When a predefined population wide dynamic range should be used.
        gt and pred will also be clipped to these values.

    Returns
    -------
    psnr : float
        Peak signal to noise ratio..

    """
    if mask is None:
        mask = np.ones(gt.shape)
    else:
        # binarize mask
        mask = np.where(mask > 0, 1., 0.)

    # Clip gt and pred to the dynamic range
    gt = np.where(gt < dynamic_range[0], dynamic_range[0], gt)
    gt = np.where(gt > dynamic_range[1], dynamic_range[1], gt)
    pred = np.where(pred < dynamic_range[0], dynamic_range[0], pred)
    pred = np.where(pred > dynamic_range[1], dynamic_range[1], pred)

    dynamic_range = dynamic_range[1] - dynamic_range[0]

    # apply mask
    gt = gt[mask == 1]
    pred = pred[mask == 1]
    psnr_value = peak_signal_noise_ratio(gt, pred, data_range=dynamic_range)
    return float(psnr_value)