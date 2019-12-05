import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import file_tool
from utils import simple_progress_bar

SingleDictionaryInstance = None

def get_dictionary_instance(*args, **key):
    global SingleDictionaryInstance
    if SingleDictionaryInstance is None:
        SingleDictionaryInstance = Dictionary(vector_length=300, test_flag=False)
    return SingleDictionaryInstance
class Dictionary:
    def __init__(self,vector_length, test_flag):
        self.vector_length = vector_length
        if test_flag:
            embedding_file = file_tool.PathManager.glove_embedding_test_file_path
            dictionary_file = file_tool.PathManager.glove_dictionary_test_file_path
        else:
            embedding_file = file_tool.PathManager.glove_embedding_file_path
            dictionary_file = file_tool.PathManager.glove_dictionary_file_path

        if not os.path.isfile(dictionary_file):
            file_operator = file_tool.FileOperator(embedding_file)
            self.dictionary = file_operator.read(self.embedding2dictionary, model='r')
            # self.dictionary = self.embedding2dictionary()
            file_tool.save_data_pickle(self.dictionary,dictionary_file)
        else:
            self.dictionary = file_tool.load_data_pickle(dictionary_file)
        pass

    def embedding2dictionary(self, word_embeddings):
        result = {}
        error_data = []
        progress_bar = simple_progress_bar.SimpleProgressBar()
        count = len(word_embeddings)
        index = 0
        while len(word_embeddings) > 0:
            index +=1
            embedding = word_embeddings[0]
            embedding = embedding.split()
            del word_embeddings[0]
            if len(embedding) != self.vector_length + 1:
                # print('vector_length error')
                error_data.append((index, ' '.join(embedding)))
                continue
            key = embedding[0]
            value = np.array(embedding[1:], dtype=str)
            value = value.astype(np.float32)
            result[key] = value
            progress_bar.update(index * 100.0 / count)

        if len(error_data) > 0:
            # error_data = pickle.dumps(error_data)
            error_data = str(error_data)
            file_tool.save_data(error_data,file_tool.PathManager.error_glove_embedding_data_file,'w')
        else:
            file_tool.save_data("No error", file_tool.PathManager.error_glove_embedding_data_file,'w')
        return result

    # def load_dictionary(self, filename):
    #     with open(filename, 'rb') as f:
    #          return pickle.load(f)
    #
    # def save_dictionary(self, filename):
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self.dictionary, f)

    def word2vector(self,word):
        find = True
        try:
            result = self.dictionary[word]
        except KeyError:
            result = np.zeros(self.vector_length)
            find = False
        return result, find



def test():
    temp = Dictionary(300,test_flag=False)


if __name__ == '__main__':
    test()