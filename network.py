
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel, DeiTModel
#from google_drive_downloader import GoogleDriveDownloader as gdd

from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation
from model.non_local import NonLocalBlock

from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn.functional as F
import math

class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        # out = torch.sign(x)
        p = (x >= 0) * (+1.0)
        n = (x < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 1.0

class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone
        self.aggregation = get_aggregation(args)
        
        self.self_att = False
        # DINOv2-large
        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 4096)
        
        # DINOv2-base
        # self.linear1 = nn.Linear(768, 768)
        # self.linear2 = nn.Linear(768, 2048)
        self.hashing = args.hashing
        if args.hashing:    
            self.aggregation_hashing = get_aggregation(args)
            self.aggregation_hashing = nn.Sequential(L2Norm(), self.aggregation_hashing, Flatten())
            # self.linear3 = nn.Linear(768, 768)
            # self.linear4 = nn.Linear(768, 512)
            self.linear3 = nn.Linear(1024, 1024)
            self.linear4 = nn.Linear(1024, 512)
        
        if args.aggregation in ["gem", "spoc", "mac", "rmac"]:
            if args.l2 == "before_pool":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
            elif args.l2 == "after_pool":
                self.aggregation = nn.Sequential(self.aggregation, L2Norm(), Flatten())
            elif args.l2 == "none":
                self.aggregation = nn.Sequential(self.aggregation, Flatten())
        
        if args.fc_output_dim != None:
            # Concatenate fully connected layer to the aggregation layer
            self.aggregation = nn.Sequential(self.aggregation,
                                             nn.Linear(args.features_dim, args.fc_output_dim),
                                             L2Norm())
            args.features_dim = args.fc_output_dim
        if args.non_local:
            non_local_list = [NonLocalBlock(channel_feat=get_output_channels_dim(self.backbone),
                                           channel_inner=args.channel_bottleneck)]* args.num_non_local
            self.non_local = nn.Sequential(*non_local_list)
            self.self_att = True

    def forward(self, x):
        x = self.backbone(x)

        if self.self_att:
            x = self.non_local(x)
        if self.arch_name.startswith("vit"):
            # return x.last_hidden_state[:, 0, :]
            B,P,D = x.last_hidden_state.shape
            W = H = int(math.sqrt(P-1))
            x1 = x.last_hidden_state[:, 1:, :].view(B,W,H,D).permute(0, 3, 1, 2) 
            x = self.aggregation(x1)
        if self.arch_name.startswith("deit"):
            x = (x.last_hidden_state[:, 0, :]+x.last_hidden_state[:, 1, :])/2.
            return x
        if self.arch_name.startswith("cct"):
            B,P,D = x.shape
            x = x.view(-1,24,24,384)
            x = x.permute(0, 3, 1, 2) 
            x = self.aggregation(x)
        if self.arch_name.startswith("dino"):
            # return x["x_norm_clstoken"]
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            if self.hashing:
                z = self.linear3(x["z_norm_patchtokens"].view(B,W,H,D)).permute(0, 3, 1, 2) 
                z = self.aggregation_hashing(z)
                z = self.linear4(z)
                z = F.normalize(z, p=2, dim=-1)
            x = self.linear1(x["x_norm_patchtokens"].view(B,W,H,D)).permute(0,3,1,2)
            x = self.aggregation(x)
            x = self.linear2(x)
        else:
            x = self.aggregation(x)

        x = F.normalize(x, p=2, dim=-1)
        if self.hashing:
            z1 = STE_binary.apply(z)
            return z, z1, x
        return x
        
def get_aggregation(args):
    if args.aggregation == "gem":
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "spoc":
        return aggregation.SPoC()
    elif args.aggregation == "mac":
        return aggregation.MAC()
    elif args.aggregation == "rmac":
        return aggregation.RMAC()
    elif args.aggregation == "netvlad":
        return aggregation.NetVLAD(clusters_num=args.netvlad_clusters, dim=args.features_dim,
                                   work_with_tokens=args.work_with_tokens)
    elif args.aggregation == 'crn':
        return aggregation.CRN(clusters_num=args.netvlad_clusters, dim=args.features_dim)
    elif args.aggregation == "rrm":
        return aggregation.RRM(args.features_dim)
    elif args.aggregation == 'none'\
            or args.aggregation == 'cls' \
            or args.aggregation == 'seqpool':
        return nn.Identity()


def get_pretrained_model(args):
    if args.pretrain == 'places':  num_classes = 365
    elif args.pretrain == 'gldv2':  num_classes = 512
    
    if args.backbone.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    elif args.backbone.startswith("resnet50"):
        model = torchvision.models.resnet50(num_classes=num_classes)
    elif args.backbone.startswith("resnet101"):
        model = torchvision.models.resnet101(num_classes=num_classes)
    elif args.backbone.startswith("vgg16"):
        model = torchvision.models.vgg16(num_classes=num_classes)
    
    if args.backbone.startswith('resnet'):
        model_name = args.backbone.split('conv')[0] + "_" + args.pretrain
    else:
        model_name = args.backbone + "_" + args.pretrain
    file_path = join("data", "pretrained_nets", model_name +".pth")
    
    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture
    args.work_with_tokens = False#args.backbone.startswith('cct') or args.backbone.startswith('vit')
    if args.backbone.startswith("resnet"):
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        elif args.backbone.startswith("resnet18"):
            backbone = torchvision.models.resnet18(pretrained=True)
        elif args.backbone.startswith("resnet50"):
            backbone = torchvision.models.resnet50(pretrained=True)
        elif args.backbone.startswith("resnet101"):
            backbone = torchvision.models.resnet101(pretrained=True)
        for name, child in backbone.named_children():
            # Freeze layers before conv_3
            if name == "layer3":
                break
            for params in child.parameters():
                params.requires_grad = False
        if args.backbone.endswith("conv4"):
            logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
            layers = list(backbone.children())[:-3]
        elif args.backbone.endswith("conv5"):
            logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
            layers = list(backbone.children())[:-2]
    elif args.backbone == "vgg16":
        if args.pretrain in ['places', 'gldv2']:
            backbone = get_pretrained_model(args)
        else:
            backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:-5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the vgg16, freeze the previous ones")
    elif args.backbone == "alexnet":
        backbone = torchvision.models.alexnet(pretrained=True)
        layers = list(backbone.features.children())[:-2]
        for l in layers[:5]:
            for p in l.parameters(): p.requires_grad = False
        logging.debug("Train last layers of the alexnet, freeze the previous ones")
    elif args.backbone.startswith("cct"):
        if args.backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation=args.aggregation)
        if args.trunc_te:
            logging.debug(f"Truncate CCT at transformers encoder {args.trunc_te}")
            backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:args.trunc_te].children())
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.classifier.blocks.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 384
        return backbone
    elif args.backbone.startswith("vit"):
        backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # if args.resize[0] == 224:
        #     backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        # elif args.resize[0] == 384:
        #     backbone = ViTModel.from_pretrained('google/vit-base-patch16-384')
        # else:
        #     raise ValueError('Image size for ViT must be either 224 or 384')

        if args.trunc_te:
            logging.debug(f"Truncate ViT at transformers encoder {args.trunc_te}")
            backbone.encoder.layer = backbone.encoder.layer[:args.trunc_te]
        if args.freeze_te:
            logging.debug(f"Freeze all the layers up to tranformer encoder {args.freeze_te+1}")
            for p in backbone.parameters():
                p.requires_grad = False
            for name, child in backbone.encoder.layer.named_children():
                if int(name) > args.freeze_te:
                    for params in child.parameters():
                        params.requires_grad = True
        args.features_dim = 768
        return backbone

    elif args.backbone.startswith("dino"):
        # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=True).eval().cuda()
        # backbone = dino
        from model.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
        # backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)  # num_register_tokens=4
        backbone = vit_large(patch_size=14,img_size=518,init_values=1,block_chunks=0)

        if not args.resume:
            model_dict = backbone.state_dict()
            # state_dict = torch.load("/home/lufeng/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
            state_dict = torch.load("/home/lufeng/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth")
            model_dict.update(state_dict.items())
            backbone.load_state_dict(model_dict)

        args.features_dim = 4096 #768#64*64#1024
        return backbone  
    
    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]

