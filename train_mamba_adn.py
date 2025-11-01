import os
from datasets import Dataset_train, Dataset_val
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import torch
import math
from Mamba_ADN_trainer import mamba_adn_trainer
import shutil
from torchvision.utils import make_grid
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=8, help='batch size')
parser.add_argument('--name', type=str, default='base', help='net name')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
parser.add_argument('--seed', type=int, default=15, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

train_data_dir = 'Mamba_syn/data/npy/train'
val_data_dir = 'Mamba_syn/data/npy/val'

save_name = 'bs{}_epoch{}_seed{}'.format(args.bs, args.epoch, args.seed)
save_dir = os.path.join('trained_models/Mamba_ADN/{}'.format(args.name), save_name)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
print(save_dir)

train_data = Dataset_train(train_dir=train_data_dir, sample_num_per_epoch=args.bs * 300)
train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)
val_data = Dataset_val(val_dir=val_data_dir)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=True, num_workers=8, pin_memory=True)

trainer = mamba_adn_trainer(max_epoch=args.epoch)
trainer.cuda()

best_MAE = 1000
print('training')
for epoch in range(args.epoch):
    epoch_train_total_loss = []
    epoch_train_ll_loss = []
    epoch_train_hh_loss = []
    epoch_train_hlh_loss = []
    epoch_train_art_loss = []
    epoch_train_GAN_lh_loss = []
    epoch_train_GAN_hl_loss = []
    epoch_train_DA_loss = []
    epoch_train_DB_loss = []
    for i, (FB_img, CT_img) in enumerate(train_dataloader):
        FB_img, CT_img = FB_img.float().cuda(), CT_img.float().cuda()
        loss_DA, loss_DB = trainer.dis_update(FB_img, CT_img)
        loss_ll, loss_hh, loss_hlh, loss_art, loss_GAN_lh, loss_GAN_hl, total_loss = trainer.gen_update(FB_img, CT_img)

        print('[%d/%d, %5d/%d] train_total_loss: %.3f train_DA_loss: %.3f train_DB_loss: %.3f train_ll_loss: %.3f '
              'train_hh_loss: %.3f train_hlh_loss: %.3f train_art_loss: %.3f train_GAN_lh_loss: %.3f train_GAN_hl_loss: %.3f'
              % (epoch + 1, args.epoch, i + 1, math.ceil(train_data.len / args.bs),
                 total_loss, loss_DA, loss_DB, loss_ll, loss_hh, loss_hlh, loss_art, loss_GAN_lh, loss_GAN_hl))
        epoch_train_total_loss.append(total_loss)
        epoch_train_ll_loss.append(loss_ll)
        epoch_train_hh_loss.append(loss_hh)
        epoch_train_hlh_loss.append(loss_hlh)
        epoch_train_art_loss.append(loss_art)
        epoch_train_GAN_lh_loss.append(loss_GAN_lh)
        epoch_train_GAN_hl_loss.append(loss_GAN_hl)
        epoch_train_DA_loss.append(loss_DA)
        epoch_train_DB_loss.append(loss_DB)
    trainer.update_learning_rate()

    epoch_val_MAE = []
    image_FB = []
    image_CT = []
    pred_CT = []
    for i, (FB_img, CT_img, mask) in enumerate(val_dataloader):
        FB_img, CT_img = FB_img.float().cuda(), CT_img.float().cuda()
        fake_CT = trainer.remove_artifacts(FB_img)

        predictions = (fake_CT.cpu().numpy() + 1) * 1250 - 1000
        real = (CT_img.cpu().numpy() + 1) * 1250 - 1000
        Mask_numpy = mask.cpu().numpy()
        MAE = (np.abs(predictions - real) * Mask_numpy).sum() / Mask_numpy.sum()
        epoch_val_MAE.append(MAE)
        if i in [2, 4, 6, 8] and epoch % 20 == 0:
            image_FB.append(FB_img[0:1, :, :, :].cpu())
            image_CT.append(CT_img[0:1, :, :, :].cpu())
            pred_CT.append(fake_CT[0:1, :, :, :].cpu())

    epoch_train_total_loss = np.mean(epoch_train_total_loss)
    epoch_train_ll_loss = np.mean(epoch_train_ll_loss)
    epoch_train_hh_loss = np.mean(epoch_train_hh_loss)
    epoch_train_hlh_loss = np.mean(epoch_train_hlh_loss)
    epoch_train_art_loss = np.mean(epoch_train_art_loss)
    epoch_train_GAN_lh_loss = np.mean(epoch_train_GAN_lh_loss)
    epoch_train_GAN_hl_loss = np.mean(epoch_train_GAN_hl_loss)
    epoch_train_DA_loss = np.mean(epoch_train_DA_loss)
    epoch_train_DB_loss = np.mean(epoch_train_DB_loss)

    epoch_val_MAE = np.mean(epoch_val_MAE)
    print(
        '[%d/%d] train_total_loss: %.3f train_DA_loss: %.3f train_DB_loss: %.3f \n'
        'val_MAE: %.3f'
        % (epoch + 1, args.epoch, epoch_train_total_loss, epoch_train_DA_loss, epoch_train_DB_loss, epoch_val_MAE))

    if epoch_val_MAE < best_MAE:
        best_MAE = epoch_val_MAE
        trainer.save(save_dir, 'best_MAE')

    train_writer.add_scalar('total_loss', epoch_train_total_loss, epoch)
    train_writer.add_scalar('ll_loss', epoch_train_ll_loss, epoch)
    train_writer.add_scalar('hh_loss', epoch_train_hh_loss, epoch)
    train_writer.add_scalar('hlh_loss', epoch_train_hlh_loss, epoch)
    train_writer.add_scalar('art_loss', epoch_train_art_loss, epoch)
    train_writer.add_scalar('GAN_lh_loss', epoch_train_GAN_lh_loss, epoch)
    train_writer.add_scalar('GAN_hl_loss', epoch_train_GAN_hl_loss, epoch)
    train_writer.add_scalar('DA_loss', epoch_train_DA_loss, epoch)
    train_writer.add_scalar('DB_loss', epoch_train_DB_loss, epoch)

    val_writer.add_scalar('MAE', epoch_val_MAE, epoch)
    val_writer.add_scalar('best_MAE', best_MAE, epoch)

    if epoch % 20 == 0:
        image_FB = torch.cat(image_FB, dim=0)
        image_CT = torch.cat(image_CT, dim=0)
        image_FB = make_grid(image_FB, 2, normalize=True)
        image_CT = make_grid(image_CT, 2, normalize=True)
        val_writer.add_image('image_FB', image_FB, epoch)
        val_writer.add_image('image_CT', image_CT, epoch)
        pred_CT = torch.cat(pred_CT, dim=0)
        pred_CT = make_grid(pred_CT, 2, normalize=True)
        val_writer.add_image('pred_CT', pred_CT, epoch)

    if (epoch + 1) == args.epoch:
        trainer.save(save_dir, 'epoch{}'.format(epoch + 1))
train_writer.close()
val_writer.close()