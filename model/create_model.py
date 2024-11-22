import os
import torch
from torch import nn
import torch.optim as optim
from .network import Detector
from .block.schedular import get_cosine_schedule_with_warmup
from .loss.focal import FocalLoss
from .loss.dice import DICELoss

def get_model(channels=128, encoder_depth=2, decoder_depth=2, neigh_size=5, margin=0.5, init_type='normal'):
    detector = Detector(channels=channels, encoder_depth=encoder_depth, decoder_depth=decoder_depth,
                        neigh_size=neigh_size, margin=margin, init_type=init_type)
    print(detector)
    return detector


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.detector = get_model(channels=opt.channels, encoder_depth=opt.encoder_depth,
                                  decoder_depth=opt.decoder_depth, neigh_size=opt.neigh_size,
                                  margin=opt.margin, init_type=opt.init_type)
        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()

        self.optimizer = optim.AdamW(self.detector.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # Here, 445 = #training images // batch_size.
        # 445 for LEVIR-CD, SECOND
        # 625 for SV-CD
        # 371 for WHU-CD
        self.schedular = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=445 * opt.warmup_epochs,
                                                         num_training_steps=445 * opt.num_epochs)
        if opt.load_pretrain:
            self.load_ckpt(self.detector, self.optimizer, opt.name, opt.backbone)
        self.detector.cuda()

        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label, boundary_mask, boundary_label):
        label = label.long()
        boundary_mask = boundary_mask.long()
        boundary_label = boundary_label.long()
        pred, ds_loss, boundary_coh_loss, boundary_cls_loss = self.detector(x1, x2, label, boundary_mask, boundary_label)
        focal = self.focal(pred, label)
        dice = self.dice(pred, label)

        return focal, dice, ds_loss, boundary_coh_loss, boundary_cls_loss

    def inference(self, x1, x2):
        with torch.no_grad():
            pred = self.detector(x1, x2)
        return  pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = '%s_%s_best.pth' % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            checkpoint = torch.load(save_path, map_location=self.device)
            network.load_state_dict(checkpoint['network'], False)

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = '%s_%s_best.pth' % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save({'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
                   save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.detector, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()

