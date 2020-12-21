import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import copy
import datasets
import models
from lib.utils import AverageMeter
from lib.protocols import *
import math
import warnings
import torch.nn.functional as F
from randaugment import RandAugmentMC
warnings.filterwarnings("ignore")

def config():
    global args
    parser = argparse.ArgumentParser(description='config for RUC')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, help='max epoch per round. (default: 200)')
    parser.add_argument('--batch_size', default=250, type=int, metavar='B', help='training batch size')
    parser.add_argument('--s_thr', default=0.99, type=float, help='confidence sampling threshold')
    parser.add_argument('--n_num', default=100, type=float, help='the number of neighbor')
    parser.add_argument('--o_model', default='checkpoint/selflabel_cifar-20.pth.tar', type=str, help='original model path')
    parser.add_argument('--e_model', default='checkpoint/simclr_cifar-20.pth.tar', type=str, help='embedding model save path')
    parser.add_argument('--seed', default=1567010775, type=int, help='random seed')
    
    args = parser.parse_args()
    return args

class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1).long(), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
LSloss = LabelSmoothLoss(0.5)

def get_threshold(current):
    return 0.9 + 0.02*int(current / 40)

def linear_rampup(current, rampup_length=200):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip((current) / rampup_length, 0.1, 1.0)
        return float(current)

class criterion_rb(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        # Clean sample Loss
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = 100*torch.mean((probs_u - targets_u)**2)
        Lu = linear_rampup(epoch) * Lu
        return Lx, Lu
            
def extract_metric(net, p_label, evalloader, n_num):
    net.eval()
    feature_bank = []
    with torch.no_grad():
        for batch_idx, (inputs1 , _, _, _, indexes) in enumerate(evalloader):
            out = net(inputs1.cuda())
            feature_bank.append(out)
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        sim_indices_list = []
        for batch_idx, (inputs1 , _, _, _, indexes) in enumerate(evalloader):
            out = net(inputs1.cuda(non_blocking=True))
            sim_matrix = torch.mm(out, feature_bank)
            _, sim_indices = sim_matrix.topk(k=n_num, dim=-1)
            sim_indices_list.append(sim_indices)
        feature_labels = p_label.cuda()
        first = True
        count = 0
        clean_num = 0
        correct_num = 0
        for batch_idx, (inputs1 , _, _, targets, indexes) in enumerate(evalloader):
            labels = p_label[indexes].cuda().long()
            sim_indices = sim_indices_list[count]
            sim_labels = torch.gather(feature_labels.expand(inputs1.size(0), -1), dim=-1, index=sim_indices)
            # counts for each class
            one_hot_label = torch.zeros(inputs1.size(0) * sim_indices.size(1), 20).cuda()
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            pred_scores = torch.sum(one_hot_label.view(inputs1.size(0), -1, 20), dim=1)
            count += 1
            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            prob, _ = torch.max(F.softmax(pred_scores, dim=-1), 1)     
            # Check whether prediction and current label are same
            noisy_label = labels
            s_idx1 = (pred_labels[:, :1].float() == labels.unsqueeze(dim=-1).float()).any(dim=-1).float()
            s_idx = (s_idx1 == 1.0)
            clean_num += labels[s_idx].shape[0]
            correct_num += torch.sum((labels[s_idx].float() == targets[s_idx].cuda().float())).item()

            if first:
                prob_set = prob
                pred_same_label_set = s_idx
                first = False
            else:
                prob_set = torch.cat((prob_set, prob), dim = 0)
                pred_same_label_set = torch.cat((pred_same_label_set, s_idx), dim = 0)

        print(correct_num, clean_num)
        return pred_same_label_set
            
def extract_confidence(net, p_label, evalloader, threshold):
    net.eval()
    devide = torch.tensor([]).cuda()
    clean_num = 0
    correct_num = 0
    for batch_idx, (inputs1, _, _, targets, indexes) in enumerate(evalloader):
        inputs1, targets = inputs1.cuda(), targets.cuda().float()
        labels = p_label[indexes].float()
        logits = net(inputs1)
        prob = torch.softmax(logits.detach_(), dim=-1)
        max_probs, _ = torch.max(prob, dim=-1)
        mask = max_probs.ge(threshold).float()
        devide = torch.cat([devide, mask])
        s_idx = (mask == 1)
        clean_num += labels[s_idx].shape[0]
        correct_num += torch.sum((labels[s_idx] == targets[s_idx])).item()
    
    print(correct_num, clean_num)
    return devide

def extract_hybrid(devide1, devide2, p_label, evalloader):
    devide = (devide1.float() + devide2.float() == 2)
    clean_num = 0
    correct_num = 0
    for batch_idx, (inputs1, _, _, targets, indexes) in enumerate(evalloader):
        inputs1, targets = inputs1.cuda(), targets.cuda().float()
        labels = p_label[indexes].float()
        mask = devide[indexes]
        s_idx = (mask == 1)
        clean_num += labels[s_idx].shape[0]
        correct_num += torch.sum((labels[s_idx] == targets[s_idx])).item()
    
    print(correct_num, clean_num)
    return devide

def preprocess(args):
    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
            transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    trainset = datasets.CIFAR20RUC(root="./data", transform=transform_test, transform2 = transform_train, transform3 = transform_train, transform4 = transform_strong, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    testset = datasets.CIFAR20RUC(root="./data",transform=transform_test, transform2 = transform_test, transform3 = transform_test,  download=False)
    evalloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    
    return trainset, trainloader, testset, evalloader, 20

def adjust_learning_rate(args, optimizer, epoch):
    # cosine learning rate schedule
    lr = args.lr
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

def train(epoch, net, net2, trainloader, optimizer, criterion_rb, devide, p_label, conf):
    train_loss = AverageMeter()
    net.train()
    net2.train()
    
    num_iter = (len(trainloader.dataset)//args.batch_size)+1
    # adjust learning rate
    adjust_learning_rate(args, optimizer, epoch)  
    optimizer.zero_grad()
    correct_u = 0
    unsupervised = 0
    conf_self = torch.zeros(50000)
    for batch_idx, (inputs1 , inputs2, inputs3, inputs4, targets, indexes) in enumerate(trainloader):
        inputs1, inputs2, inputs3, inputs4, targets = inputs1.float().cuda(), inputs2.float().cuda(), inputs3.float().cuda(), inputs4.float().cuda(), targets.cuda().long()
        s_idx = (devide[indexes] == 1)
        u_idx = (devide[indexes] == 0)
        labels = p_label[indexes].cuda().long()
        labels_x = torch.tensor(p_label[indexes][s_idx]).squeeze().long().cpu()
        target_x = torch.zeros(labels_x.shape[0], 20).scatter_(1, labels_x.view(-1,1), 1).float().cuda()
        
        logit_o, logit_w1, logit_w2, logit_s = net(inputs1), net(inputs2), net(inputs3), net(inputs4)
        logit_s = logit_s[s_idx]
        max_probs, _ = torch.max(torch.softmax(logit_o, dim=1), dim=-1)
        conf_self[indexes] = max_probs.detach().cpu()
        optimizer.zero_grad()
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u11 = logit_w1[u_idx]
            outputs_u21  = logit_w2[u_idx]
            logit_o2 = net2(inputs1)
            logit_w12 = net2(inputs2)
            logit_w22 = net2(inputs3)
            outputs_u12 = logit_w12[u_idx]
            outputs_u22  = logit_w22[u_idx]
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
            ptu = pu**(1/0.5) # temparature sharpening
            target_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            target_u = target_u.detach().float() 
            
            px = torch.softmax(logit_o2[s_idx], dim=1)
            w_x = conf[indexes][s_idx]
            w_x = w_x.view(-1,1).float().cuda() 
            px = (1-w_x)*target_x + w_x*px              
            ptx = px**(1/0.5) # temparature sharpening           
            target_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            target_x = target_x.detach().float()      
            
            if logit_o[u_idx].shape[0] > 0: 
                max_probs, targets_u1 = torch.max(torch.softmax(logit_o[u_idx], dim=1), dim=-1)
                thr = get_threshold(epoch)    
                mask_u = max_probs.ge(thr).float()
                u_idx2 = (mask_u == 1)
                unsupervised += torch.sum(mask_u).item()
                correct_u += torch.sum((targets_u1[u_idx2] == targets[u_idx][u_idx2])).item()
                update = indexes[u_idx][u_idx2]
                devide[update] = True
                p_label[update] = targets_u1[u_idx2].float()
        
        
        l = np.random.beta(4.0, 4.0)        
        l = max(l, 1-l)
        
        all_inputs = torch.cat([inputs2[s_idx], inputs3[s_idx], inputs2[u_idx], inputs3[u_idx]],dim=0)
        all_targets = torch.cat([target_x, target_x, target_u, target_u], dim=0)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        batch_size = target_x.shape[0]
        
        Lx, Lu = criterion_rb(logits[:batch_size*2], mixed_target[:batch_size*2], logits[batch_size*2:], mixed_target[batch_size*2:], epoch+batch_idx/num_iter)
        total_loss = Lx + Lu + LSloss(logit_s, labels_x.cuda())
        
        total_loss.backward()
        train_loss.update(total_loss.item(), inputs2.size(0))
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Epoch: [{epoch}][{elps_iters}/{tot_iters}] '
                  'Train loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '.format(
                      epoch=epoch, elps_iters=batch_idx,tot_iters=len(trainloader), 
                      train_loss=train_loss))
    conf_self = (conf_self - conf_self.min()) / (conf_self.max() - conf_self.min())
    return train_loss.avg, devide, p_label, conf_self

            
def main():
    args = config()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainset, trainloader, testset, evalloader, class_num = preprocess(args)
    net = models.ClusteringModel(models.__dict__['Resnet_CIFAR'](), class_num)
    net2 = copy.deepcopy(net)
    net_uc = copy.deepcopy(net)
    net_embd = models.ContrastiveModel(models.__dict__['Resnet_CIFAR']())
    try:
        state_dict = torch.load(args.o_model)
        state_dict2 = torch.load(args.e_model)
        net_uc.load_state_dict(state_dict)
        net_embd.load_state_dict(state_dict2, strict = True)
        net.load_state_dict(state_dict, strict = False)
        net2.load_state_dict(state_dict, strict = False)
        net.cluster_head = nn.ModuleList([nn.Linear(512, class_num) for _ in range(1)])
        net2.cluster_head = nn.ModuleList([nn.Linear(512, class_num) for _ in range(1)])
    except:
        print("Check Model Directory!")
        exit(0)
        
    if args.device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
        net_uc = torch.nn.DataParallel(net_uc, device_ids=range(torch.cuda.device_count()))
        net_embd = torch.nn.DataParallel(net_embd, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    net.to(args.device)
    net2.to(args.device)
    net_uc.to(args.device)
    net_embd.to(args.device)
    
    optimizer1 = torch.optim.SGD(net.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer2 = torch.optim.SGD(net2.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    criterion = criterion_rb()
    
    # Extract Pseudo Label
    acc_uc, p_label= test(net_uc, evalloader, args.device, class_num)
    print(acc_uc)
    devide1 = extract_confidence(net_uc, p_label, evalloader, args.s_thr)
    devide2 = extract_metric(net_embd, p_label, evalloader, args.n_num)
    devide = extract_hybrid(devide1, devide2, p_label, evalloader)
    
    conf1 =  torch.zeros(50000)
    conf2 =  torch.zeros(50000)
    
    for epoch in range(args.epochs):    
        print("== Train RUC ==")
        loss, devide, p_label, conf1 = train(epoch, net, net2, trainloader, optimizer1, criterion, devide, p_label, conf2)
        loss, devide, p_label, conf2 = train(epoch, net2, net, trainloader, optimizer2, criterion, devide, p_label, conf1)
        acc, p_list = test_ruc(net, net2, evalloader, args.device, class_num)
        print("accuracy: {}\n".format(acc))
        
        state = {'net1': net.state_dict(),
                 'net2': net2.state_dict() }
        torch.save(state, './checkpoint/ruc_cifar20.t7')
        
if __name__ == "__main__":
    main()