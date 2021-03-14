from utils import *
from preprocess import Preprocess
from data import ReviewDataset
from main import model_dir, testing_data_path, w2v_path, device, path_prefix, label_map, sen_len
from test import testing

print("loading testing data ...")
testing_data = load_data_unlabled(testing_data_path)
tokens_test = tokenization(testing_data)

# we don`t need to set a small batch here~
#
batch_size=256

preprocess = Preprocess(tokens_test, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
tokens_test = preprocess.sentence_word2idx()
test_dataset = ReviewDataset(X=tokens_test, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 0)

print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'ckpt.model'))
outputs = testing(batch_size, test_loader, model, device)
outputs = labels_norm(outputs, label_map, norm=False)

# write into the files
print('\nsaving prediction result')
with open(path_prefix+'/data/test_prediction.txt', mode='w+', encoding='utf-8') as f:
    for i in outputs:
        f.write(i+'\n')

with open(path_prefix+'/data/test_prediction_comparation.txt', mode='w+', encoding='utf-8') as f:
    for i in range(len(outputs)):
        text = testing_data[i] + ';' + outputs[i] + '\n'
        f.write(text)
print("Finish Predicting")