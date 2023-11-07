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
        lmd=1,
        bidir=False,
        reg_weights=None
    ):
        super().__init__()
        dim = len(vol_size)
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
    
    def training_step(self, batch, batch_idx):
        fixed = batch['fixed']
        moving = batch['moving']
        fixed_map = batch['fixed_map']
        moving_map = batch['moving_map']

        fixed = fixed.permute(0, 3, 1, 2)
        moving = moving.permute(0, 3, 1, 2)

        # preprocess label map
        fixed_map = F.one_hot(fixed_map, num_classes=self.num_labels)
        fixed_map = fixed_map.permute(0, 3, 1, 2)
        fixed_map = fixed_map.contiguous().type(torch.float32)
        
        moving_map = F.one_hot(moving_map, num_classes=self.num_labels)
        moving_map = moving_map.permute(0, 3, 1, 2)
        moving_map = moving_map.contiguous().type(torch.float32)

        results = self.reg_model(moving, fixed, False)

        y_source, y_target, flow = results['y_source'], results['y_target'], results['flow']
        pred = layers.SpatialTransFormer('linear', 'ij', fill_value=0)([moving_map, flow])
        dice_loss = self.dice_loss.loss(fixed_map, pred) + 1.
        grad_loss = self.l2_loss.loss(None, flow)
        self.log_dict(
            dictionary={
                'dice_loss': dice_loss, 
                'grad_loss': grad_loss
            }, 
            on_epoch=True,
            on_step=False, 
            prog_bar=True,
        )

        return dice_loss + grad_loss
    
    
    def predict_step(self, moving, fixed):
        fixed = fixed.permute(0, 3, 1, 2)
        moving = moving.permute(0, 3, 1, 2)
        results = self.reg_model(moving, fixed, True)
        flow = results["flow"]
        moved = results["y_source"]
        return moved, flow

    
    def configure_optimizers(self, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer