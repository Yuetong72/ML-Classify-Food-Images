import numpy as np
import torch.nn as nn
import torch

def update_learning_rate(opt,decay_rate=0.8,min_value=1e-3):
    for pg in opt.param_groups:
        pg["lr"]=max(pg["lr"]*decay_rate,min_value)
    print("learning rate",pg["lr"])

def save_model(model,filename,args):
    if args.ngpu > 1:
        torch.save(model.module.state_dict(), filename.replace(".pth",'_params.pth'))
        # torch.save(model.module, filename)
    else:
        torch.save(model.state_dict(), filename.replace(".pth", '_params.pth'))
        # torch.save(model, filename)

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()
