import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from config import Config
from utils.dataloader import TestDataset


def inference(datasets):
    global model, cfg
    model.eval()
    for dataset in datasets:
        assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K', 'DUTS', 'DUT', 'ECSSD', 'HKU_IS']
        save_path = os.path.join('prediction_maps', dataset)
        os.makedirs(save_path, exist_ok=True)

        test_dataset = TestDataset(image_root=getattr(cfg.dp, f'test_{dataset}_imgs'),
                                   gt_root=getattr(cfg.dp, f'test_{dataset}_masks'),
                                   testsize=cfg.trainsize)

        for img, gt, gt_origin, name in tqdm(test_dataset):
            img = img.unsqueeze(0).cuda()
            out1 = model(img)[0]
            out1 = F.interpolate(out1, size=gt_origin.shape[1:], mode='bilinear', align_corners=True)
            out1 = torch.sigmoid(out1) * 255
            out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            # save preds
            cv2.imwrite(os.path.join(save_path, name), out1)


if __name__ == '__main__':
    # set path to pth
    pth_path = 'save_pth/epoch_100.pth'

    cfg = Config()

    from Model.SAENet_R2N50 import SAENet
    # from Model.SAENet_PVTv2 import SAENet
    model = SAENet(channel=64).to(cfg.device)

    model.load_state_dict(torch.load(pth_path))

    datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
    # datasets = ['DUTS', 'DUT', 'ECSSD', 'HKU_IS']
    inference(datasets)