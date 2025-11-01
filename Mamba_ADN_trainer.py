import torch
import torch.nn as nn
import os
from Networks.Mamba_ADN_nets import ADN, NLayerDiscriminator

class mamba_adn_trainer(nn.Module):
    def __init__(self, max_epoch):
        super(mamba_adn_trainer, self).__init__()
        self.model_g = ADN(input_ch=1, base_ch=64, num_down=2, num_residual=2, unm_mamba=2, num_sides=3,
                 res_norm='instance', down_norm='instance', up_norm='layer', fuse=True, shared_decoder=False)
        self.model_dl = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=2, norm_layer='instance')
        self.model_dh = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=2, norm_layer='instance')

        self.model_g_opt = torch.optim.Adam(self.model_g.parameters(),
                                        lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        self.model_dl_opt = torch.optim.Adam(self.model_dl.parameters(),
                                          lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
        self.model_dh_opt = torch.optim.Adam(self.model_dh.parameters(),
                                          lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)

        # self.model_g_scheduler = torch.optim.lr_scheduler.StepLR(self.model_g_opt, step_size=1.e+5, gamma=0.5)
        # self.model_dl_scheduler = torch.optim.lr_scheduler.StepLR(self.model_dl_opt, step_size=1.e+5, gamma=0.5)
        # self.model_dh_scheduler = torch.optim.lr_scheduler.StepLR(self.model_dh_opt, step_size=1.e+5, gamma=0.5)
        lambda_lr = lambda epoch: (1 - epoch / max_epoch) ** 0.9
        self.model_g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.model_g_opt, lr_lambda=lambda_lr)
        self.model_dl_scheduler = torch.optim.lr_scheduler.LambdaLR(self.model_dl_opt, lr_lambda=lambda_lr)
        self.model_dh_scheduler = torch.optim.lr_scheduler.LambdaLR(self.model_dh_opt, lr_lambda=lambda_lr)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def MSE_loss(self, input, target=0):
        return torch.mean((input - target)**2)

    def MAE_loss(self, input, target=0):
        return torch.mean(torch.abs(input - target))

    def update_learning_rate(self):
        self.model_g_scheduler.step()
        self.model_dl_scheduler.step()
        self.model_dh_scheduler.step()

    def dis_update(self, img_low, img_high):
        self.model_dl_opt.zero_grad()
        self.model_dh_opt.zero_grad()
        self.model_g.eval()
        self.model_dl.train()
        self.model_dh.train()

        pred_ll, pred_lh = self.model_g.forward1(img_low)
        pred_hl, pred_hh = self.model_g.forward2(img_low, img_high)

        loss_real_l = self.MSE_loss(self.model_dl(img_low), target=1)
        loss_fake_l = self.MSE_loss(self.model_dl(pred_hl), target=0)
        loss_Dl = (loss_real_l + loss_fake_l) / 2
        loss_Dl.backward()
        torch.nn.utils.clip_grad_norm_(self.model_dl.parameters(), 12)
        self.model_dl_opt.step()

        loss_real_h = self.MSE_loss(self.model_dh(img_high), target=1)
        loss_fake_h = self.MSE_loss(self.model_dh(pred_lh), target=0)
        loss_Dh = (loss_real_h + loss_fake_h) / 2
        loss_Dh.backward()
        self.model_dh_opt.step()
        torch.nn.utils.clip_grad_norm_(self.model_dh.parameters(), 12)
        return loss_Dl.item(), loss_Dh.item()

    def gen_update(self, img_low, img_high):
        self.model_g_opt.zero_grad()
        self.model_g.train()
        self.model_dl.eval()
        self.model_dh.eval()

        pred_ll, pred_lh = self.model_g.forward1(img_low)
        pred_hl, pred_hh = self.model_g.forward2(img_low, img_high)

        # pred_lhl = self.model_g.forward_hl(pred_hl, pred_lh)
        pred_hlh = self.model_g.forward_lh(pred_hl)

        loss_GAN_lh = self.MSE_loss(self.model_dh(pred_lh), target=1)
        loss_GAN_hl = self.MSE_loss(self.model_dl(pred_hl), target=1)

        loss_ll = self.recon_criterion(pred_ll, img_low)
        loss_hh = self.recon_criterion(pred_hh, img_high)
        loss_hlh = self.recon_criterion(pred_hlh, img_high)
        loss_art = self.recon_criterion(img_low - pred_lh, pred_hl - img_high)

        total_loss = 20 * loss_ll + 20 * loss_hh + 20 * loss_hlh + 20 * loss_art + loss_GAN_lh + loss_GAN_hl
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model_g.parameters(), 12)
        self.model_g_opt.step()
        return loss_ll.item(), loss_hh.item(), loss_hlh.item(), loss_art.item(), \
               loss_GAN_lh.item(), loss_GAN_hl.item(), total_loss.item()

    def remove_artifacts(self, img_low):
        self.model_g.eval()
        with torch.no_grad():
            pred_hlh = self.model_g.forward_lh(img_low)
        return pred_hlh

    def save(self, save_dir, description):
        Gen_name = os.path.join(save_dir, 'Gen_{}.pth'.format(description))
        Dis_A_name = os.path.join(save_dir, 'Dis_A_{}.pth'.format(description))
        Dis_B_name = os.path.join(save_dir, 'Dis_B_{}.pth'.format(description))
        torch.save(self.model_g.state_dict(), Gen_name)
        torch.save(self.model_dl.state_dict(), Dis_A_name)
        torch.save(self.model_dh.state_dict(), Dis_B_name)
        print('save: {} {}'.format(save_dir, description))

    def load(self, model_dir, description):
        Gen_name = os.path.join(model_dir, 'Gen_{}.pth'.format(description))
        Dis_A_name = os.path.join(model_dir, 'Dis_A_{}.pth'.format(description))
        Dis_B_name = os.path.join(model_dir, 'Dis_B_{}.pth'.format(description))
        self.model_g.load_state_dict(torch.load(Gen_name))
        self.model_dl.load_state_dict(torch.load(Dis_A_name))
        self.model_dh.load_state_dict(torch.load(Dis_B_name))
