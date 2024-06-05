import os
import torch


class Config:
    def __init__(self):
        dataset_dir = './data'
        self.dp = DataPath(dataset_dir)
        self.num_workers = 4

        self.CUDA = True
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        # training settings
        self.epochs = 100
        self.trainsize = 384
        # self.batch_size = 20
        self.batch_size = 12
        self.weight_decay = 5e-4
        self.learning_rate = 1e-4
        self.lr_step_size = 10
        self.lr_decay_rate = 0.5

        # loss function
        from utils.loss import structure_loss, FocalLossWithLogits
        self.mask_loss = structure_loss
        self.edge_loss = FocalLossWithLogits(alpha=0.5, gamma=2)


class DataPath:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        '''COD'''
        self.cod_dataset_dir = os.path.join(dataset_dir, 'COD')
        # CAMO-Train + COD10K-Train
        self.train_imgs = os.path.join(self.cod_dataset_dir, 'TrainDataset', 'Imgs')
        self.train_masks = os.path.join(self.cod_dataset_dir, 'TrainDataset', 'GT')
        self.train_edges = os.path.join(self.cod_dataset_dir, 'TrainDataset', 'Edge')
        # CHAMELEON
        self.test_CHAMELEON_imgs = os.path.join(self.cod_dataset_dir, 'TestDataset', 'CHAMELEON', 'Imgs')
        self.test_CHAMELEON_masks = os.path.join(self.cod_dataset_dir, 'TestDataset', 'CHAMELEON', 'GT')
        # CAMO-Test
        self.test_CAMO_imgs = os.path.join(self.cod_dataset_dir, 'TestDataset', 'CAMO', 'Imgs')
        self.test_CAMO_masks = os.path.join(self.cod_dataset_dir, 'TestDataset', 'CAMO', 'GT')
        # COD10K-Test
        self.test_COD10K_imgs = os.path.join(self.cod_dataset_dir, 'TestDataset', 'COD10K', 'Imgs')
        self.test_COD10K_masks = os.path.join(self.cod_dataset_dir, 'TestDataset', 'COD10K', 'GT')
        # NC4K
        self.test_NC4K_imgs = os.path.join(self.cod_dataset_dir, 'TestDataset', 'NC4K', 'Imgs')
        self.test_NC4K_masks = os.path.join(self.cod_dataset_dir, 'TestDataset', 'NC4K', 'GT')

        '''SOD'''
        self.sod_dataset_dir = os.path.join(dataset_dir, 'SOD')
        #  DUTS-TR
        self.train_DUTS_imgs = os.path.join(self.sod_dataset_dir, 'DUTS-TR', 'DUTS-TE-Image')
        self.train_DUTS_masks = os.path.join(self.sod_dataset_dir, 'DUTS-TR', 'DUTS-TE-Mask')
        self.train_DUTS_edges = os.path.join(self.sod_dataset_dir, 'DUTS-TR', 'DUTS-TE-Edge')
        #  DUTS-TE
        self.test_DUTS_imgs = os.path.join(self.sod_dataset_dir, 'DUTS-TE', 'DUTS-TE-Image')
        self.test_DUTS_masks = os.path.join(self.sod_dataset_dir, 'DUTS-TE', 'DUTS-TE-Mask')
        #  DUT-OMRON
        self.test_DUT_imgs = os.path.join(self.sod_dataset_dir, 'DUT-OMRON', 'Test', 'Image')
        self.test_DUT_masks = os.path.join(self.sod_dataset_dir, 'DUT-OMRON', 'Test', 'GT')
        # ECSSD
        self.test_ECSSD_imgs = os.path.join(self.sod_dataset_dir, 'ECSSD', 'Test', 'Image')
        self.test_ECSSD_masks = os.path.join(self.sod_dataset_dir, 'ECSSD', 'Test', 'GT')
        # HKU-IS
        self.test_HKU_IS_imgs = os.path.join(self.sod_dataset_dir, 'HKU-IS', 'Test', 'Image')
        self.test_HKU_IS_masks = os.path.join(self.sod_dataset_dir, 'HKU-IS', 'Test', 'GT')
