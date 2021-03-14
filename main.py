# main.py
# To go through the whole process,
# including:
# 1.Data Cleaning
# 2.Embedding
# 3.Preprocessing
# 4.Model Design and Training
# 5.Evaluation

import os
import torch
from sklearn import model_selection
from utils import load_data_labled, load_data_unlabled, labels_norm, tokenization
from data import ReviewDataset
from preprocess import Preprocess
from model import LSTM_Net
from train import training

label_map = {'anger': 0, 'fear': 1, 'sadness': 2, 'love': 3, 'joy': 4, 'surprise': 5}
path_prefix = './'

# check if we can use gpu to do the job
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data path
train_x_path = os.path.join(path_prefix, 'data/train.txt')
valid_x_path = os.path.join(path_prefix, 'data/val.txt')
testing_data_path = os.path.join(path_prefix, 'data/test_data.txt')

# w2v model path
w2v_path = os.path.join(path_prefix, 'model/w2v_all.model')  # 處理 word to vec model 的路徑

# lstm model path
model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model


# set some essential parameters
# sen_len 30: because finally i dont want to delete the stopwords.
# batch_size 32: small batch size can make the optimizer more randomly and can get out the local point!
# fix embedding during training, here we dont want to train them together
fix_embedding = True
sen_len = 30
batch_size = 32
epoch = 40
lr = 0.001

if __name__ == '__main__':
    # take the raw data in
    train_x, train_labels = load_data_labled(train_x_path)
    valid_x, valid_labels =load_data_labled(valid_x_path)
    testing_data = load_data_unlabled(testing_data_path)

    # tokenization, DO NOT delete the stopwords
    tokens_train = tokenization(train_x + valid_x, del_stop=False)
    tokens_test = tokenization(testing_data, del_stop=False)

    # normalize the labels in int type
    train_labels = labels_norm(train_labels, label_map)
    valid_labels = labels_norm(valid_labels, label_map)

    # transform the tokens into id as well as get the embedding matrix
    preprocess = Preprocess(tokens_train, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    labels = preprocess.labels_to_tensor(train_labels+valid_labels)

    # establish the model instance
    model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
    model = model.to(device)


    # generate training set and validation set
    X_train, X_val, y_train, y_val = model_selection.train_test_split(train_x, labels, test_size=0.2, random_state=233)

    # dataloader to implement the batch training
    train_dataset = ReviewDataset(X=X_train, y=y_train)
    val_dataset = ReviewDataset(X=X_val, y=y_val)
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True,
                                                num_workers = 0)

    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = False,
                                                num_workers = 0)

    # start training!!!
    training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)