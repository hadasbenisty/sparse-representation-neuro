import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import utils
import Model
import Simulate_Dataset
import sys
import os
import scipy.io as spio

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


if __name__ == '__main__':
    input_dir   = sys.argv[1]
    T           = int(sys.argv[2])        # number of encoder unfolding
    L           = int(sys.argv[3])        # 1/L is the step size in the encoder
    C           = int(sys.argv[4])        # number of filters
    K           = int(sys.argv[5])        # length of the filters
    lam         = float(sys.argv[6])      # regularization weight
    lr          = float(sys.argv[7])      # learning rate
    batch_size  = int(sys.argv[8])        # batch size
    num_epochs  = int(sys.argv[9])        # epocs
    
    # mkdir, set output file, check if it exists
    outputfolder = 'results/' + input_dir + '/'
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    outputfile = outputfolder + "results_CRsAE1D_T" + str(T) + "_L" + str(L) + "_C" + str(C) + "_K" + str(K) + "_lam" + str(lam) + "_lr" + str(lr) + "_bs" + str(batch_size) + "_ep" + str(num_epochs)  
    
    if not os.path.exists(outputfile + ".mat"):       
            # define network parameters
        net_hyp = {"T": T,                 # number of encoder unfolding
                   "L": L,                  # 1/L is the step size in the encoder
                   "C": C,                 # number of filters
                   "K": K,                  # length of the filters
                   "lam": lam,
                   "lr": lr,
                   "batch_size": batch_size,
                   "device": device
                   }
    
        net = Model.CRsAE1D(net_hyp)
    
        # training parameters
        train_hyp = {"batch_size": batch_size, "num_epochs": num_epochs, "lr": lr, "shuffle": True}
        dataset = Simulate_Dataset.LoadDataset(r'data/' + input_dir + '/data.mat', device=device)
        train_loader = DataLoader(dataset, shuffle=train_hyp["shuffle"], batch_size=train_hyp["batch_size"])
        # criterion
        criterion = torch.nn.MSELoss()
        # optimizer
        optimizer = optim.Adam(net.parameters(), lr=train_hyp["lr"], eps=1e-3)
    
    
        ### train the model ###
        for epoch in tqdm(range(train_hyp["num_epochs"]), disable=True):
            loss_all = 0
            for idx, y in tqdm(enumerate(train_loader), disable=False):
                y = y.to(device)
                y_hat, _ = net(y)
                loss = criterion(y, y_hat)
                loss_all += float(loss.item())
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                net.normalize()
    
            print("Epoch [{}/{}]loss:{:.4f}\n".format(epoch+1, train_hyp["num_epochs"], loss_all))
    
        #MSE_error = utils.evaluate_model(dataset, net, device, criterion)
        #print(f'The MSE of the model from the true value is {MSE_error}')
    
        yi = dataset[:]
        yi_hat, xi_hat = net(yi)
        h = net.get_param("H")
        #utils.vis_data_est(yi[1, 0][0:500], yi_hat[1, 0][0:500])
        #utils.vis_data(xi_hat[1, 0][0:500])
        #utils.vis_filters(net.get_param("H"))
    
        # save the trained model in data folder
        #torch.save(net.state_dict(), outputfile + '/model')
        spio.savemat(outputfile + ".mat" ,
             {'yi_hat': yi_hat, 'yi': yi, 'h': h, 'xi_hat': xi_hat})
