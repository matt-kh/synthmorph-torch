import pytorch_lightning as pl
import torch
# local code
from .networks import *
from .losses import *


class SynthMorph(pl.LightningModule):
    def __init__(
        self,
        vol_size,
        num_labels,
        enc_nf,
        dec_nf,
        int_steps=7,
        int_downsize=2,
        bidir=False,
        lmd=1,
        lr=1e-4,
        reg_weights=None,
       
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vol_size = vol_size
        self.num_labels = num_labels
        self.reg_model = VxmDense(
            inshape=vol_size,
            nb_unet_features=[enc_nf, dec_nf],
            int_steps=int_steps,
            int_downsize=int_downsize,
            bidir=bidir,
            unet_half_res=True,
        )
        if reg_weights is not None:
            self.reg_model.load_state_dict(torch.load(reg_weights))   # .pth file

        self.dice_loss = Dice()
        self.l2_loss = Grad(penalty='l2', loss_mult=lmd)
        self.lmd = lmd
        
        self.lr = lr
    
    def training_step(self, batch):
        fixed = batch['fixed']
        moving = batch['moving']
        fixed_map = batch['fixed_map']
        moving_map = batch['moving_map']

        results = self.reg_model(moving, fixed)

        y_source, y_target, warp = results['y_source'], results['y_target'], results['flow']
        pred = layers.SpatialTransformer(fill_value=0)([moving_map, warp])
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