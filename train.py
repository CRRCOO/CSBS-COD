import torch
import random
import numpy as np
from tqdm import tqdm
import os

from config import Config
from utils.dataloader import TrainDataset
from utils.tools import clip_gradient


def train():
    global model, train_datald, optimizer, cfg, scheduler
    for epoch in range(cfg.epochs):
        model.train()

        loss_iter = []
        for img, gt, edge in tqdm(train_datald):
            optimizer.zero_grad()

            img = img.to(cfg.device)
            gt = gt.to(cfg.device)
            edge = edge.to(cfg.device)

            out, c1, c2, c3, c4, c5, e1, e2, e3, e4, e5 = model(img)
            oloss = cfg.mask_loss(out, gt)
            closs1 = cfg.mask_loss(c1, gt)
            closs2 = cfg.mask_loss(c2, gt)
            closs3 = cfg.mask_loss(c3, gt)
            closs4 = cfg.mask_loss(c4, gt)
            closs5 = cfg.mask_loss(c5, gt)
            eloss1 = cfg.edge_loss(e1, edge)
            eloss2 = cfg.edge_loss(e2, edge)
            eloss3 = cfg.edge_loss(e3, edge)
            eloss4 = cfg.edge_loss(e4, edge)
            eloss5 = cfg.edge_loss(e5, edge)
            closs = closs1 + closs2 + closs3 + closs4 + closs5
            eloss = eloss1 + eloss2 + eloss3 + eloss4 + eloss5
            loss = oloss + closs + eloss

            loss.backward()
            clip_gradient(optimizer, grad_clip=0.5)
            optimizer.step()

            loss_iter.append(loss.item())

        print(f'Epoch: {epoch + 1}, LR: {np.round(scheduler.get_last_lr(), 8)}, Loss: {np.round(np.mean(loss_iter), 8)}')
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == cfg.epochs - 1:
            torch.save(model.state_dict(), f'save_pth/epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    cfg = Config()

    from Model.SAENet_R2N50 import SAENet
    # from Model.SAENet_PVTv2 import SAENet
    model = SAENet(channel=64).to(cfg.device)

    save_dir = './save_pth'
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = TrainDataset(image_root=cfg.dp.train_imgs, gt_root=cfg.dp.train_masks, trainsize=cfg.trainsize,
                                 edge_root=cfg.dp.train_edges)
    train_datald = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_decay_rate)

    train()
