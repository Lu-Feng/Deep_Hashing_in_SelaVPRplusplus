
import math
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler,autocast

import util
import parser
import commons
import datasets_ws
from model import network
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint
import os 
import warnings
warnings.filterwarnings("ignore")
import test_rerank
    
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

#### Creation of Datasets
logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

# triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train", args.negs_num_per_query)
# logging.info(f"Train query set: {triplets_ds}")

val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, "pitts30k", "test")
logging.info(f"Test set: {test_ds}")

#### Initialize model
model = network.GeoLocalizationNet(args)
model = model.to(args.device)

if args.resume:
    model = util.resume_model(args, model)
best_r5 = start_epoch_num = not_improved_num = 0

model = torch.nn.DataParallel(model)

adapters_state = model.module.backbone.adapters.state_dict()
model.module.backbone.adapters_hashing.load_state_dict(adapters_state)

for name, param in model.module.named_parameters():
    if "adapters_hashing" in name or "linear3" in name or "linear4" in name or "aggregation_hashing" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
for name, param in model.module.named_parameters():
    if param.requires_grad == True:
        print(name)
 
#### Setup Optimizer and Loss
if args.aggregation == "crn":
    crn_params = list(model.module.aggregation.crn.parameters())
    net_params = list(model.module.backbone.parameters()) + \
        list([m[1] for m in model.module.aggregation.named_parameters() if not m[0].startswith('crn')])
    if args.optim == "adam":
        optimizer = torch.optim.Adam([{'params': crn_params, 'lr': args.lr_crn_layer},
                                      {'params': net_params, 'lr': args.lr_crn_net}])
        logging.info("You're using CRN with Adam, it is advised to use SGD")
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD([{'params': crn_params, 'lr': args.lr_crn_layer, 'momentum': 0.9, 'weight_decay': 0.001},
                                     {'params': net_params, 'lr': args.lr_crn_net, 'momentum': 0.9, 'weight_decay': 0.001}])
else:
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

if args.criterion == "triplet":
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")
elif args.criterion == "sare_ind":
    criterion_triplet = sare_ind
elif args.criterion == "sare_joint":
    criterion_triplet = sare_joint

if torch.cuda.device_count() >= 2:
    # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
    model = convert_model(model)
    model = model.cuda()

from torchvision import transforms as T
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

TRAIN_CITIES = [
    'SFXL',
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL', # refers to Oslo
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT', # refers to Toronto
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG', # refers to Prague
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS', # refers to Paris
]

batch_size=args.train_batch_size #32
img_per_place=4
min_img_per_place=4
shuffle_all=False
image_size=(224, 224)#(256,256)#(320,320)#
num_workers=4
cities=TRAIN_CITIES
mean_std=IMAGENET_MEAN_STD
random_sample_from_each_place=True

mean_dataset = mean_std['mean']
std_dataset = mean_std['std']
train_transform = T.Compose([
    T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
    T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=mean_dataset, std=std_dataset),
])

train_loader_config = {
    'batch_size': batch_size,
    'num_workers': num_workers,
    'drop_last': False,
    'pin_memory': True,
    'shuffle': shuffle_all}

train_dataset = GSVCitiesDataset(
            cities=cities,
            img_per_place=img_per_place,
            min_img_per_place=min_img_per_place,
            random_sample_from_each_place=random_sample_from_each_place,
            transform=train_transform)

from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=CosineSimilarity())
#  The loss function call (this method will be called at each training iteration)
def loss_function(descriptors, labels):
    # we mine the pairs/triplets if there is an online mining strategy
    if miner is not None:
        miner_outputs = miner(descriptors, labels)
        # print(loss_fn(descriptors, labels))
        # print(len(set(miner_outputs[0].detach().cpu().numpy())),len(set(miner_outputs[1].detach().cpu().numpy())),len(set(miner_outputs[2].detach().cpu().numpy())),len(set(miner_outputs[3].detach().cpu().numpy())))
        loss = loss_fn(descriptors, labels, miner_outputs)
        # print(loss)

        # calculate the % of trivial pairs/triplets 
        # which do not contribute in the loss value
        nb_samples = descriptors.shape[0]
        nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
        batch_acc = 1.0 - (nb_mined/nb_samples)

    else: # no online mining
        loss = loss_fn(descriptors, labels)
        batch_acc = 0.0
    return loss, miner_outputs

def gaussian(x, mu=0, sigma=1):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

#### Training loop
ds = DataLoader(dataset=train_dataset, **train_loader_config)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=4000*5)
scaler = GradScaler()
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")
    
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0,1), dtype=np.float32)
    
    model = model.train()
    epoch_losses=[]
    for images, place_id in tqdm(ds):
        BS, N, ch, h, w = images.shape
        # reshape places and labels
        images = images.view(BS*N, ch, h, w)
        labels = place_id.view(-1)

        optimizer.zero_grad()
        with autocast():

            float_descriptors, quant_descriptors, rerank_descriptors = model(images.to(args.device))
            quant_descriptors = quant_descriptors.cuda()
            loss, miner_outputs = loss_function(quant_descriptors, labels) # Call the loss_function we defined above 
            index1 = torch.randperm(len(miner_outputs[0]))[:max(int(1.0*len(miner_outputs[0])),1)] #Change "1.0" to a number in (0,1), to randomly select a portion of the positive-pairs.
            index2 = torch.randperm(len(miner_outputs[2]))[:max(int(1.0*len(miner_outputs[2])),1)] #for negative-pairs
            sim_posi = torch.sum(float_descriptors[miner_outputs[0][index1]] * float_descriptors[miner_outputs[1][index1]],dim=-1)
            sim_nega = torch.sum(float_descriptors[miner_outputs[2][index2]] * float_descriptors[miner_outputs[3][index2]],dim=-1)
            bin_sim_posi = torch.sum(quant_descriptors[miner_outputs[0][index1]] * quant_descriptors[miner_outputs[1][index1]],dim=-1) / quant_descriptors.shape[-1]
            bin_sim_nega = torch.sum(quant_descriptors[miner_outputs[2][index2]] * quant_descriptors[miner_outputs[3][index2]],dim=-1) / quant_descriptors.shape[-1]
            loss2 = torch.mean(torch.cat([(sim_posi - bin_sim_posi).pow(2),(sim_nega - bin_sim_nega).pow(2)],dim=0))
            print(loss,loss2)
            loss = loss+0.1*loss2
            del quant_descriptors, float_descriptors, rerank_descriptors

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # scheduler.step()       
        
        # Keep track of all losses by appending them to epoch_losses
        batch_loss = loss.item()
        epoch_losses = np.append(epoch_losses, batch_loss)
        del loss
        
        # logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
        #               f"current batch triplet loss = {batch_loss:.4f}, " +
        #               f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                 f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test_rerank.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[0]+recalls[1] > best_r5
    
    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": best_r5,
        "not_improved_num": not_improved_num
    }, is_best, filename="last_model.pth")
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {(recalls[0]+recalls[1]):.1f}")
        best_r5 = (recalls[0]+recalls[1])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {(recalls[0]+recalls[1]):.1f}")
        if not_improved_num >= args.patience:
            logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

#### Test best model on test set
best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test_rerank.test(args, test_ds, model, test_method=args.test_method)
logging.info(f"Recalls on {test_ds}: {recalls_str}")