import torch
import torch.nn as nn
from models.hd3net import HD3Net
from hd3losses import *
from utils.visualizer import get_visualization


class HD3Model(nn.Module):

    def __init__(self, task, encoder, decoder, corr_range=None, context=False):
        super(HD3Model, self).__init__()
        self.ds = 6  # default downsample ratio of the coarsest level
        self.task = task
        self.encoder = encoder
        self.decoder = decoder
        self.corr_range = corr_range
        self.context = context
        self.criterion = LossCalculator(task)
        self.eval_epe = EndPointError
        self.hd3net = HD3Net(task, encoder, decoder, corr_range, context,
                             self.ds)

    def forward(self,
                img_list,
                label_list=None,
                get_vect=True,
                get_prob=False,
                get_loss=False,
                get_epe=False,
                get_auc=False,
                get_vis=False):
        result = {}

        ms_prob, ms_vect = self.hd3net(torch.cat(img_list, 1))
        if get_vect:
            result['vect'] = ms_vect[-1]
        if get_prob:
            result['prob'] = ms_prob[-1]
        if get_loss:
            result['loss'] = self.criterion(ms_prob, ms_vect, label_list[0], self.corr_range, self.ds)
        if get_epe:
            scale_factor = 1 / 2**(self.ds - len(ms_vect) + 1)
            res = self.eval_epe(ms_vect[-1] * scale_factor, label_list[0])
            if isinstance(res, tuple): # KITTI dataset - evaluate epe-all, epe-noc, error rate [%]
                result['epe'] = res[0]
                result['epe_noc'] = res[1]
                result['er'] = res[2]
            else:
                result['epe'] = res

        if get_auc:
            scale_factor = 1 / 2**(self.ds - len(ms_vect) + 1)
            auc, rel_auc = evaluate_auc(ms_prob[-1].data, ms_vect[-1]*scale_factor, label_list[0])
            result['auc'] = auc
            result['rel_auc'] = rel_auc

        if get_vis:
            result['vis'] = get_visualization(img_list, label_list, ms_vect,
                                              ms_prob, self.ds)

        return result
