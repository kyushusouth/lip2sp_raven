import torch
from raven_copy import  E2E
import hydra
import sys


@hydra.main(config_name="resnet_transformer_base", config_path="../conf/model/visual_backbone")
def main(cfg):
    ckpt_path = '/home/minami/raven_data/raven_vox2lrs3_base_video.pth'
    # ckpt_path = '/home/minami/raven_data/raven_vox2lrs3_large_video.pth'
    pretrained_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
    raven = E2E(1000, cfg).encoder
    model_dict = raven.state_dict()
    match_dict = {name: params for name, params in pretrained_dict.items() if name in model_dict}
    raven.load_state_dict(match_dict, strict=True)

    for name, param in raven.named_parameters():
        if not (torch.equal(param, pretrained_dict[name])):
            print(name)

    


if __name__ == '__main__':
    main()