import os
import numpy as np
import torch
import argparse
import shutil
from Nii_utils import NiiDataRead, NiiDataWrite
from Mamba_ADN_trainer import mamba_adn_trainer
from metrics_utils import compute_mae, ssim, psnr
import pandas as pd
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--model_dir', type=str, help='trained model dir')
parser.add_argument('--set', type=str, help='val/test')
parser.add_argument('--model_mode', type=str, default='best_MAE', help='best_MAE or best_dice')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

if args.set == 'val':
    test_data_dir = 'data/data20240613/val'
elif args.set == 'test':
    test_data_dir = 'data/data20240613/test'
name_list = os.listdir(test_data_dir)
HU_clip = [-1000, 1500]

new_dir = os.path.join(args.model_dir, args.model_mode + '_{}'.format(args.set))
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
print(new_dir)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)
metrics_all = {'ID': [],
               'CBCT_MAE_Body': [], 'CBCT_SSIM_Body': [], 'CBCT_PSNR_Body': [],
               'CBCT_MAE_CTV': [], 'CBCT_SSIM_CTV': [], 'CBCT_PSNR_CTV': [],
               'CBCT_MAE_Heart': [], 'CBCT_SSIM_Heart': [], 'CBCT_PSNR_Heart': [],
               'CBCT_MAE_Lungleft': [], 'CBCT_SSIM_Lungleft': [], 'CBCT_PSNR_Lungleft': [],
               'CBCT_MAE_Lungright': [], 'CBCT_SSIM_Lungright': [], 'CBCT_PSNR_Lungright': [],
               'CBCT_MAE_PCTV': [], 'CBCT_SSIM_PCTV': [], 'CBCT_PSNR_PCTV': [],
               'CBCT_MAE_Spinalcord': [], 'CBCT_SSIM_Spinalcord': [], 'CBCT_PSNR_Spinalcord': [],
               'CBCT_MAE_Stomach': [], 'CBCT_SSIM_Stomach': [], 'CBCT_PSNR_Stomach': [],
               'sCT_MAE_Body': [], 'sCT_SSIM_Body': [], 'sCT_PSNR_Body': [],
               'sCT_MAE_CTV': [], 'sCT_SSIM_CTV': [], 'sCT_PSNR_CTV': [],
               'sCT_MAE_Heart': [], 'sCT_SSIM_Heart': [], 'sCT_PSNR_Heart': [],
               'sCT_MAE_Lungleft': [], 'sCT_SSIM_Lungleft': [], 'sCT_PSNR_Lungleft': [],
               'sCT_MAE_Lungright': [], 'sCT_SSIM_Lungright': [], 'sCT_PSNR_Lungright': [],
               'sCT_MAE_PCTV': [], 'sCT_SSIM_PCTV': [], 'sCT_PSNR_PCTV': [],
               'sCT_MAE_Spinalcord': [], 'sCT_SSIM_Spinalcord': [], 'sCT_PSNR_Spinalcord': [],
               'sCT_MAE_Stomach': [], 'sCT_SSIM_Stomach': [], 'sCT_PSNR_Stomach': []}

trainer = mamba_adn_trainer(max_epoch=200)
trainer.cuda()

trainer.load(model_dir=args.model_dir, description=args.model_mode)

for i, name in enumerate(name_list):
    print(i, name)
    CBCT_img, spacing, origin, direction = NiiDataRead(os.path.join(test_data_dir, name, 'CBCT.nii.gz'))
    CT_img, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'CT.nii.gz'))
    Body_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'mask.nii.gz'))
    Body_mask[Body_mask > 0] = 1
    for roi in ['CTV', 'Heart', 'Lungleft', 'Lungright', 'PCTV', 'Spinalcord', 'Stomach']:
        try:
            exec("{}_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, '{}.nii.gz'))".format(roi, roi))
            exec("{}_mask[{}_mask > 0] = 1".format(roi, roi))
        except:
            exec("{}_mask = None".format(roi))
    # Heart_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'Heart.nii.gz'))
    # Lungleft_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'Lungleft.nii.gz'))
    # Lungright_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'Lungright.nii.gz'))
    # PCTV_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'PCTV.nii.gz'))
    # Spinalcord_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'Spinalcord.nii.gz'))
    # Stomach_mask, _, _, _ = NiiDataRead(os.path.join(test_data_dir, name, 'Stomach.nii.gz'))
    # CTV_mask[CTV_mask > 0] = 1
    # Heart_mask[Heart_mask > 0] = 1
    # Lungleft_mask[Lungleft_mask > 0] = 1
    # Lungright_mask[Lungright_mask > 0] = 1
    # PCTV_mask[PCTV_mask > 0] = 1
    # Spinalcord_mask[Spinalcord_mask > 0] = 1
    # Stomach_mask[Stomach_mask > 0] = 1

    # CBCT_img[Body_mask == 0] = HU_clip[0]
    CBCT_img = np.clip(CBCT_img, HU_clip[0], HU_clip[1])
    FB_img_norm = (CBCT_img - HU_clip[0]) / (HU_clip[1] - HU_clip[0]) * 2 - 1

    # CT_img[Body_mask == 0] = HU_clip[0]
    CT_img = np.clip(CT_img, HU_clip[0], HU_clip[1])
    CT_img_norm = (CT_img - HU_clip[0]) / (HU_clip[1] - HU_clip[0]) * 2 - 1

    pred_CT = np.zeros(CT_img.shape)

    with torch.no_grad():
        for n in range(FB_img_norm.shape[0]):
            FB_img_norm_one = FB_img_norm[n]
            FB_img_norm_one = torch.from_numpy(FB_img_norm_one).float().unsqueeze(0).unsqueeze(0).cuda()
            CT_img_norm_one = CT_img_norm[n]
            CT_img_norm_one = torch.from_numpy(CT_img_norm_one).float().unsqueeze(0).unsqueeze(0).cuda()
            pred_CT_one = trainer.remove_artifacts(FB_img_norm_one)
            pred_CT[n] = pred_CT_one.squeeze(0).squeeze(0).cpu().numpy()

    pred_CT = (pred_CT + 1) * 1250 - 1000
    pred_CT[Body_mask == 0] = HU_clip[0]

    NiiDataWrite(os.path.join(new_dir, 'predictions', '{}.nii.gz'.format(name)),
                 pred_CT, spacing, origin, direction)

    metrics_all['ID'].append(name)
    for roi in ['Body', 'CTV', 'Heart', 'Lungleft', 'Lungright', 'PCTV', 'Spinalcord', 'Stomach']:
        try:
            exec('CBCT_MAE_{} = compute_mae(CBCT_img, CT_img, mask={}_mask)'.format(roi, roi))
            exec('CBCT_SSIM_{} = ssim(CT_img, CBCT_img, {}_mask, HU_clip)'.format(roi, roi))
            exec('CBCT_PSNR_{} = psnr(CT_img, CBCT_img, {}_mask, HU_clip)'.format(roi, roi))
            exec('sCT_MAE_{} = compute_mae(pred_CT, CT_img, mask={}_mask)'.format(roi, roi))
            exec('sCT_SSIM_{} = ssim(CT_img, pred_CT, {}_mask, HU_clip)'.format(roi, roi))
            exec('sCT_PSNR_{} = psnr(CT_img, pred_CT, {}_mask, HU_clip)'.format(roi, roi))
        except:
            exec('CBCT_MAE_{} = 0'.format(roi))
            exec('CBCT_SSIM_{} = 0'.format(roi))
            exec('CBCT_PSNR_{} = 0'.format(roi))
            exec('sCT_MAE_{} = 0'.format(roi))
            exec('sCT_SSIM_{} = 0'.format(roi))
            exec('sCT_PSNR_{} = 0'.format(roi))
        exec("metrics_all['CBCT_MAE_{}'].append(CBCT_MAE_{})".format(roi, roi))
        exec("metrics_all['CBCT_SSIM_{}'].append(CBCT_SSIM_{})".format(roi, roi))
        exec("metrics_all['CBCT_PSNR_{}'].append(CBCT_PSNR_{})".format(roi, roi))
        exec("metrics_all['sCT_MAE_{}'].append(sCT_MAE_{})".format(roi, roi))
        exec("metrics_all['sCT_SSIM_{}'].append(sCT_SSIM_{})".format(roi, roi))
        exec("metrics_all['sCT_PSNR_{}'].append(sCT_PSNR_{})".format(roi, roi))

    # CBCT_MAE_Body = compute_mae(CBCT_img, CT_img, mask=Body_mask)
    # CBCT_SSIM_Body = ssim(CT_img, CBCT_img, Body_mask, HU_clip)
    # CBCT_PSNR_Body = psnr(CT_img, CBCT_img, Body_mask, HU_clip)
    # sCT_MAE_Body = compute_mae(pred_CT, CT_img, mask=Body_mask)
    # sCT_SSIM_Body = ssim(CT_img, pred_CT, Body_mask, HU_clip)
    # sCT_PSNR_Body = psnr(CT_img, pred_CT, Body_mask, HU_clip)
    # metrics_all['CBCT_MAE_Body'].append(CBCT_MAE_Body)
    # metrics_all['CBCT_SSIM_Body'].append(CBCT_SSIM_Body)
    # metrics_all['CBCT_PSNR_Body'].append(CBCT_PSNR_Body)
    # metrics_all['sCT_MAE_Body'].append(sCT_MAE_Body)
    # metrics_all['sCT_SSIM_Body'].append(sCT_SSIM_Body)
    # metrics_all['sCT_PSNR_Body'].append(sCT_PSNR_Body)

    print('{}: {} {} {} {} {} {}'.format(name, CBCT_MAE_Body, CBCT_SSIM_Body, CBCT_PSNR_Body, sCT_MAE_Body, sCT_SSIM_Body, sCT_PSNR_Body))

metrics_all['ID'].append('mean')
metrics_all['ID'].append('std')
print('mean:')
for metric in ['CBCT_MAE_Body', 'CBCT_SSIM_Body', 'CBCT_PSNR_Body',
               'CBCT_MAE_CTV', 'CBCT_SSIM_CTV', 'CBCT_PSNR_CTV',
               'CBCT_MAE_Heart', 'CBCT_SSIM_Heart', 'CBCT_PSNR_Heart',
               'CBCT_MAE_Lungleft', 'CBCT_SSIM_Lungleft', 'CBCT_PSNR_Lungleft',
               'CBCT_MAE_Lungright', 'CBCT_SSIM_Lungright', 'CBCT_PSNR_Lungright',
               'CBCT_MAE_PCTV', 'CBCT_SSIM_PCTV', 'CBCT_PSNR_PCTV',
               'CBCT_MAE_Spinalcord', 'CBCT_SSIM_Spinalcord', 'CBCT_PSNR_Spinalcord',
               'CBCT_MAE_Stomach', 'CBCT_SSIM_Stomach', 'CBCT_PSNR_Stomach',
               'sCT_MAE_Body', 'sCT_SSIM_Body', 'sCT_PSNR_Body',
               'sCT_MAE_CTV', 'sCT_SSIM_CTV', 'sCT_PSNR_CTV',
               'sCT_MAE_Heart', 'sCT_SSIM_Heart', 'sCT_PSNR_Heart',
               'sCT_MAE_Lungleft', 'sCT_SSIM_Lungleft', 'sCT_PSNR_Lungleft',
               'sCT_MAE_Lungright', 'sCT_SSIM_Lungright', 'sCT_PSNR_Lungright',
               'sCT_MAE_PCTV', 'sCT_SSIM_PCTV', 'sCT_PSNR_PCTV',
               'sCT_MAE_Spinalcord', 'sCT_SSIM_Spinalcord', 'sCT_PSNR_Spinalcord',
               'sCT_MAE_Stomach', 'sCT_SSIM_Stomach', 'sCT_PSNR_Stomach']:
    metrics_one = metrics_all[metric]
    metrics_one = [x for x in metrics_one if x != 0]
    mean = np.nanmean(metrics_one)
    std = np.nanstd(metrics_one)
    metrics_all[metric].append(mean)
    metrics_all[metric].append(std)
    print('{}: {}'.format(metric, mean))
frame = pd.DataFrame(metrics_all)
frame.to_csv(os.path.join(new_dir, 'metrics.csv'), index=False)
frame.to_csv(os.path.join(new_dir, 'metrics.csv'), index=False)