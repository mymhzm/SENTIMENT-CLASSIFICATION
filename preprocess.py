# feature generation
import torch
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path='./model/w2v.model'):
        self.w2v_path = w2v_path
        self.sentences = sentences

        # set a fixed sentence length, our lstm model can not handle the variable sequence
        self.sen_len = sen_len

        self.word2idx = {}  # id-word
        self.idx2word = [] # word-id
        self.embedding_matrix = [] # vector -> matrix

    def get_w2v_model(self):
        # load the trained w2v model
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # in order to add a new words to the embedding
        # for example the <PAD> and <UNK>
        vector = torch.empty(1, self.embedding_dim)
        # vector plus the weights
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        ## the whole word2vec embedding
        ## prepare for the model embedding layer
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.vocab): # .wv.vocab can get the dictionary which is generated in the w2v model training
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # add two special str words
        # pad is used to padding the short sentence
        # unk is used to replace the words not in the dict
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # make all the sentence the same length
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            # not enough , we can add a <PAD>
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        return sentence

    def sentence_word2idx(self):
        # change the str into int id,
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)