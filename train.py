# train.py
# the training details and save some evaluation scores in each epoch
import numpy as np
from torch import nn
import torch
import torch.optim as optim
from utils import evaluation

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    # set the cost function
    criterion = nn.CrossEntropyLoss()
    t_batch = len(train)
    v_batch = len(valid)
    # set the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_loss = 0, 0, 999

    # intermediate evaluation score
    epoch_ev = []
    train_loss_ev = []
    train_ac_ev = []
    test_loss_ev = []
    test_ac_ev = []

    for epoch in range(n_epoch):
        epoch_ev.append(epoch+1)
        total_loss, total_acc = 0, 0
        # a epoch training with many batch
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.long)
            # clean up the grad from last batch
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            # update the parameters W
            optimizer.step()

            # calculate the accuracy
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	    epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r') # end='\r'不换行输出 default:end='\n'
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        train_loss_ev.append(total_loss/t_batch)
        train_ac_ev.append(total_acc/t_batch*100)

        # a epoch validation
        model.eval()
        # without grad graph to save the resource
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            test_loss_ev.append(total_loss / v_batch)
            test_ac_ev.append(total_acc / v_batch * 100)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))

            # save the besst model
            if total_loss < best_loss:
                best_loss = total_loss
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))

        print('-----------------------------------------------')
        model.train()

    # save the epoch evaluation
    ev_path = './evaluation/'
    np.save(ev_path+'epoch_ev', np.array(epoch_ev))
    np.save(ev_path+'train_loss_ev', np.array(train_loss_ev))
    np.save(ev_path+'train_ac_ev', np.array(train_ac_ev))
    np.save(ev_path+'test_loss_ev', np.array(test_loss_ev))
    np.save(ev_path+'test_ac_ev', np.array(test_ac_ev))

