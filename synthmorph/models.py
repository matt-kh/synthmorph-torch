import pytorch_lightning as pl
import torch
# local code
from .networks import *
from .losses import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class SynthMorph(pl.LightningModule):
    def __init__(
        self,
        vol_size,
        num_labels,
        enc_nf,
        dec_nf,
        int_steps=7,
        svf_resolution=2,
        int_resolution=2,
        bidir=False,
        lmd=1,
        lr=1e-4,
        reg_weights=None,
        device=device,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vol_size = vol_size
        self.num_labels = num_labels
        self.reg_model = VxmDense(
            inshape=vol_size,
            nb_unet_features=[enc_nf, dec_nf],
            int_steps=int_steps,
            svf_resolution=svf_resolution,
            int_resolution=int_resolution,
            bidir=bidir,
        )
        self.reg_model = self.reg_model.to(device=device)
        if reg_weights is not None:
            self.reg_model.load_state_dict(torch.load(reg_weights))   # .pth file

        self.dice_loss = Dice()
        self.l2_loss = Grad(penalty='l2', loss_mult=lmd)
        self.lmd = lmd
        
        self.lr = lr
    
    def training_step(self, batch):
        moving = batch['moving']
        fixed = batch['fixed']
        moving_map = batch['moving_map']
        fixed_map = batch['fixed_map']

        results = self.reg_model(moving, fixed)

        y_source, y_target, warp = results['y_source'], results['y_target'], results['flow']
        pred = layers.SpatialTransformer(fill_value=0)([torch_to_tf(moving_map), torch_to_tf(warp)])
        pred = tf_to_torch(pred)
        dice_loss = self.dice_loss.loss(fixed_map, pred) + 1.
        grad_loss = self.l2_loss.loss(None, warp)
    
        self.log_dict(
            dictionary={
                'dice_loss': dice_loss, 
                'grad_loss': grad_loss,
                'total_loss': dice_loss + grad_loss
            }, 
            on_epoch=True,
            on_step=False, 
            prog_bar=True,
        )

        return dice_loss + grad_loss
    
    
    def predict_step(self, moving, fixed):
        results = self.reg_model(moving, fixed)
        flow = results["flow"]
        moved = results["y_source"]
        return moved, flow

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def save_weigths(self, pth):
        torch.save(self.reg_model.state_dict(), pth)


class SynthMorphAffine(pl.LightningModule):
    def __init__(
        self,
        vol_size,
        enc_nf=[256] * 4,
        dec_nf=[256] * 0,
        add_nf=[256] * 4,
        lr=1e-4,
        reg_weights=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.vol_size = vol_size
        self.reg_model = VxmAffineFeatureDetector(
            in_shape=vol_size,
            enc_nf=enc_nf,
            dec_nf=dec_nf,
            add_nf=add_nf,
            bidir=True  # remove later
        )
        self.transformer = layers.SpatialTransformer(fill_value=0)
        if reg_weights is not None:
            self.reg_model.load_state_dict(torch.load(reg_weights))   # .pth file

        self.mse_loss = MSE()
        self.lr = lr
    
    def training_step(self, batch):
        moving = batch['moving']
        fixed = batch['fixed']
        moving_map = batch['moving_map']
        fixed_map = batch['fixed_map']
        results = self.reg_model(moving, fixed)
        trans = results['aff_1']
        # if torch.isnan(trans).any():
        #     print(f"NaN values detected in the trans.")
        pred = self.transformer([torch_to_tf(moving_map), torch_to_tf(trans)])
        pred = tf_to_torch(pred)
        mse_loss = self.mse_loss.loss(fixed_map, pred)
        # if torch.isnan(mse_loss).any():
        #     print(f"NaN values detected in the mse.")

    
        self.log_dict(
            dictionary={
                'mse_loss': mse_loss
            }, 
            on_epoch=True,
            on_step=True, 
            prog_bar=True,
        )

        return mse_loss
    
    
    def predict_step(self, moving, fixed):
        results = self.reg_model(moving, fixed)
        trans = results["aff_1"]
        moved = self.transformer([torch_to_tf(moving), torch_to_tf(trans)])
        moved = tf_to_torch(moved)
        return moved, trans
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def save_weigths(self, pth):
        torch.save(self.reg_model.state_dict(), pth)