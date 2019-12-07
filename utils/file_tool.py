import os
import pickle

def load_data(file_name, model):
    with open(file_name, model) as f:
        rows = f.readlines()
    return rows


def save_data(data, file_name, model):
    with open(file_name, model) as f:
        f.write(data)


def save_data_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_data_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class FileOperator:
    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, operation_func, model):
        result = load_data(self.file_name,model)
        return operation_func(result)

    def write(self, all_data, operation_func, model):
        all_data = operation_func(all_data)
        save_data(all_data,self.file_name, model)

class PathManager:
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(base_dir, 'data')

    glove_path = os.path.join(data_path, 'glove')
    glove_embedding_file_path = os.path.join(glove_path, 'glove.840B.300d.txt')
    glove_embedding_test_file_path = os.path.join(glove_path, 'glove.840B.300d_test.txt')
    glove_dictionary_file_path = os.path.join(glove_path, 'glove_dictionary.pkl')
    glove_dictionary_test_file_path = os.path.join(glove_path, 'glove_dictionary_test.pkl')



    msrpc_path = os.path.join(data_path, 'msrpc')
    msrpc_train_data_set_path = os.path.join(msrpc_path, 'train.txt')
    msrpc_test_data_set_path = os.path.join(msrpc_path, 'test.txt')
    msrpc_train_data_manager_path = os.path.join(msrpc_path, 'train_data_manager.pkl')
    msrpc_test_data_manager_path = os.path.join(msrpc_path, 'test_data_manager.pkl')

    result_path = os.path.join(base_dir, 'result')
    log_path = os.path.join(result_path,'log')
    error_glove_embedding_data_file = os.path.join(log_path, 'error_glove_embedding_data.txt')
    word_no_in_dictionary_file = os.path.join(log_path, 'word_no_in_dictionary')

    model_path = os.path.join(result_path,'model.txt')
    entire_model_file =  os.path.join(model_path, 'entire_model.pkl')


    def change_filename_by_append(self, file_path, append_str):
        (dir_path, file_name_ext) = os.path.split(file_path)
        (filename, extension) = os.path.splitext(file_name_ext)
        new_path = dir_path+r'\r'+filename + append_str +extension
        return new_path
