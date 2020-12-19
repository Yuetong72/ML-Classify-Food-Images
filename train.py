from __future__ import print_function
import argparse
import torch.optim as optim
from AI.utils import *
import numpy as np
import os
import time
from torchvision import transforms
from AI.efficientnet.model import EfficientNet


defaultencoding = 'utf-8'

# Training settings

parser = argparse.ArgumentParser(description='PyTorch EfficientNet Training for iFood-Rice')
parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training (default: 32)')
parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 5e-4)')
parser.add_argument('--resume_path', type=str, default=r"", help='checkpoint path for resuming')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--log_interval', type=int, default=300, help='how many batches to wait before logging training status')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--freeze', type=int, default=0, help='number of layers to freeze (default: 0)')
# parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}

# preprocessing

transform_train = transforms.Compose([
                                transforms.Resize([224,224],3),
                                transforms.RandomAffine(degrees=360,translate=(0.2,0.2),scale=(0.8,1.2),shear=0,resample=3),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ColorJitter(0.3, 0.3, 0.3, 0),
                                transforms.ToTensor()])
                                
transform_test = transforms.Compose([transforms.Resize([224,224],3),
                                transforms.ToTensor()])

# training and validation set

train_dataset = ImageList("/content/train_set",r'./data/train_info.csv',transform=transform_train,image_mode="RGB",loader=pil_loader)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,num_workers=args.workers)
test_dataset = ImageList("/content/val_set",r'./data/val_info.csv',transform=transform_train,image_mode="RGB",loader=pil_loader)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,num_workers=args.workers)
test_list = test_dataset.data

# load pretrain model

device = torch.device("cuda" if args.cuda else "cpu")
model=EfficientNet.from_name("efficientnet-b2", override_params={'num_classes':251})
state_dict=torch.load("efficientnet-b2-8bb594d6.pth")
# model=EfficientNet.from_name("efficientnet-b4", override_params={'num_classes':251})
# state_dict=torch.load("efficientnet-b4-6ed6700e.pth")
state_dict.pop('_fc.weight')
state_dict.pop('_fc.bias')
model.load_state_dict(state_dict, strict=False)
model=model.to(device)

# freeze parameters for first few layers

ct = 0
for child in model.children():
    ct += 1
    if ct < args.freeze:
        for param in child.parameters():
            param.requires_grad = False

print("total layer:" + str(ct))


# loss function

criterion=nn.CrossEntropyLoss().to(device)

# load checkpoint

if len(args.resume_path)!=0:
    if args.resume_path.endswith("_params.pth"):
        data_dict = torch.load(args.resume_path)
        model.load_state_dict(data_dict)
        model.to(device)

# Parallelization of GPU

if args.ngpu>1:
    model = torch.nn.DataParallel(model).cuda()

# Optimizer

# optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=args.lr, momentum=args.momentum,weight_decay=5e-4)
optimizer=optim.Adam(model.parameters(),lr=args.lr)

# training

def train(epoch):
    model.train()
    correct=0
    Loss=[]
    t_start = time.time()
    epoch_start_time=time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if isinstance(output, tuple):
            output = output[0]
            
        target=target.view(-1).long()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1)[1] # get the index of the max log-probability
        correct_batch=pred.eq(target).cpu().sum()
        correct += pred.eq(target).cpu().sum().item()
        Loss.append(loss.item())
        # measure elapsed time
        batch_time=time.time() - t_start
        t_start = time.time()
        acc_batch=100. * correct_batch / len(target.data)
        train_loss_batch=loss.item()
        if batch_idx % args.log_interval == 0:
            print_log('Epoch: [{}][{}/{}]    batch_time: {:.3f}   loss: {:.4f}    acc: {:.3f}%'.format(
                epoch, batch_idx*args.train_batch_size, len(train_loader.dataset), batch_time,train_loss_batch,acc_batch),log)
    acc = 100. * correct / (len(train_loader.dataset))
    print("Train acc,",acc)
    train_acc_all.append(acc)
    train_loss_all.append(np.mean(Loss))
    print_log('Time for an Epoch: {} \n'.format(time.time() - epoch_start_time),log)
        
def test(epoch):
    model.eval()
    correct=0
    Loss=[]
    t_start = time.time()
    epoch_start_time=time.time()
    errors=[]
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data, target = data.to(device),target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            target=target.view(-1).long()
            loss = criterion(output, target)

            pred = output.max(1)[1] # get the index of the max log-probability

            pred_top3=output.argsort(dim=1)[:,-3:].cpu().numpy()
            gt=target.cpu().numpy()
            for i in range(pred_top3.shape[0]):
                if gt[i] in pred_top3[i,:]:
                    errors.append(0)
                else:
                    errors.append(1)

            correct_batch=pred.eq(target).cpu().sum()
            correct += pred.eq(target).cpu().sum().item()
            Loss.append(loss.item())
            # measure elapsed time
            batch_time=time.time() - t_start
            t_start = time.time()

    acc = 100. * correct / len(test_loader.dataset)
    error =np.mean(errors)
    print("Test acc,",acc)
    print("error",error)
    test_acc_all.append(acc)
    test_loss_all.append(np.mean(Loss))
    test_error_all.append(error)
    global best_acc,save_weight_dir,best_error
    if acc>best_acc:
        save_model(model,os.path.join(save_weight_dir, 'epoch_%d_acc_%.4f.pth'%(epoch,acc)),args)
        best_acc=acc

    if error<best_error:
        save_model(model,os.path.join(save_weight_dir, 'epoch_%d_error_%.4f.pth'%(epoch,error)),args)
        best_error=error

if __name__ == '__main__':
    train_acc_all = []
    test_acc_all = []
    train_loss_all = []
    test_loss_all = []
    test_error_all = []
    best_acc = 0
    best_error = 1
    root_dir = time.strftime("results_classification", time.localtime())
    save_weight_dir = os.path.join(root_dir, 'weights_best')
    if not os.path.exists(save_weight_dir):
        os.makedirs(save_weight_dir)
    save_figures_dir = os.path.join(root_dir, 'figures')
    if not os.path.exists(save_figures_dir):
        os.makedirs(save_figures_dir)

    log = open(os.path.join(save_weight_dir, 'print_log.txt'), 'w')
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)

        update_learning_rate(optimizer,decay_rate=0.8,min_value=1e-7)
        
        plot_error_curve(train_acc_all, test_acc_all, ylabel='Accuracy',save_dir=save_figures_dir, save_filename='Accuracy_curve')
        plot_error_curve(train_loss_all, test_loss_all, ylabel='Loss', save_dir=save_figures_dir,
                        save_filename='Loss_curve')
        plot_error_curve(test_error_all, test_error_all, ylabel='Error', save_dir=save_figures_dir,
                        save_filename='Error_curve')
    log.close()
    #'''

