import torch
import numpy as np

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def DBindex(cl_data_file):
    class_list = cl_data_file.keys()
    cl_num= len(class_list)
    cl_means = []
    stds = []
    DBs = []
    for cl in class_list:
        cl_means.append( np.mean(cl_data_file[cl], axis = 0) )
        stds.append( np.sqrt(np.mean( np.sum(np.square( cl_data_file[cl] - cl_means[-1]), axis = 1))))

    mu_i = np.tile( np.expand_dims( np.array(cl_means), axis = 0), (len(class_list),1,1) )
    mu_j = np.transpose(mu_i,(1,0,2))
    mdists = np.sqrt(np.sum(np.square(mu_i - mu_j), axis = 2))
    
    for i in range(cl_num):
        DBs.append( np.max([ (stds[i]+ stds[j])/mdists[i,j]  for j in range(cl_num) if j != i ]) )
    return np.mean(DBs)

def sparsity(cl_data_file):
    class_list = cl_data_file.keys()
    cl_sparsity = []
    for cl in class_list:
        cl_sparsity.append(np.mean([np.sum(x!=0) for x in cl_data_file[cl] ])  ) 

    return np.mean(cl_sparsity) 

def kl_diag_gauss_with_standard_gauss(mean, logvar):
    mean_flat = torch.cat([t.view(-1) for t in mean])
    logvar_flat = torch.cat([t.view(-1) for t in logvar])
    var_flat = logvar_flat.exp()

    return -0.5 * torch.sum(1 + logvar_flat - mean_flat.pow(2) - var_flat)

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu
