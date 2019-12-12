from abc import abstractmethod
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import file_tool
import torch.utils.data as torch_data
import numpy as np
import re
import torch
import os
import random
import utils.word_embedding as word_embedding
from utils import word_embedding

class DataItem:
    def __init__(self, original_data):
        self.original_data = original_data
        self.convert_data_form()

    @abstractmethod
    def convert_data_form(self):
        pass

    @abstractmethod
    def get_tensor(self):
        pass

# countttt =0
# nocountttt =0
class MSRPInput(DataItem):
    def convert_data_form(self):
        global nocountttt
        word_dictionary = word_embedding.get_dictionary_instance()
        self.no_find_word_list = []
        self.no_find_word_dict = {}
        self.remove_default_word_vector = False
        if type(self.original_data) != str:
            raise ValueError
        new_sentence = []
        for i,word in enumerate(self.original_data.split()):
            r=re.finditer(r'[^a-zA-Z]|[a-zA-Z]+',word)
            for part in r:
                substr = word[part.span()[0] : part.span()[1]]
                # if len(re.findall(r'[a-zA-Z]',substr))>1 and substr
                #     raise ValueError
                vector, find_flag = word_dictionary.word2vector(substr)
                new_sentence.append(vector)
                if not find_flag:
                    # nocountttt +=1
                    self.no_find_word_list.append((word, vector))
                    self.no_find_word_dict[word] = vector
        self.sentence = np.array(new_sentence, dtype=np.float)
        self.sentence_length = len(self.sentence)

    def align_sentence(self, length):
        shape = self.sentence.shape
        word_dictionary = word_embedding.get_dictionary_instance()
        dim = 0
        for i, size in enumerate(shape):
            if size != word_dictionary.vector_length:
                dim = i
        shape = list(shape)
        if(length - shape[dim] <0):
            return
        shape[dim] = length - shape[dim]
        temp = torch.zeros(*shape)
        self.sentence = np.concatenate([self.sentence, temp], axis=dim)
        # print(self.sentence)

    def get_tensor(self, use_gpu):
        result = torch.from_numpy(self.sentence.T).type(torch.float32)
        if use_gpu:
            result = result.cuda()
        return result

    def remove_word_not_in_embedding_dictionary(self):
        self.remove_default_word_vector -=self.remove_default_word_vector
        default_vector = word_embedding.get_dictionary_instance().default_vector()
        temp = []
        for  i in range(len(self.sentence)):
            if np.all(self.sentence[i] == default_vector):
                temp.insert(0,i)
        for index in temp:
            self.sentence = np.delete(self.sentence, index, axis=0)
        self.sentence_length = len(self.sentence)
        # print(self.sentence )


class MSRPLabel(DataItem):
    def convert_data_form(self):
        self.label = int(self.original_data)
        if self.label != 0 and self.label != 1:
            raise ValueError

    def get_tensor(self, use_gpu):
        result = torch.tensor(self.label, dtype=torch.uint8)
        if use_gpu:
            result =result.cuda()
        return result


class DataElement:
    def __init__(self, **items):
        self.members = items

    def __getitem__(self, item):
        return self.members[item]

    def __setitem__(self, key, value):
        self.members[key] = value

    def __iter__(self):
        return self


class DataExample(DataElement):
    @abstractmethod
    def get_example_pair(self):
        pass


class MSRPDataExample(DataExample):
    def __init__(self, input_sentence1:MSRPInput, input_sentence2:MSRPInput, label:MSRPLabel):
        super().__init__(input_sentence1=input_sentence1, input_sentence2=input_sentence2, label=label)
        self.input_align_length = -1

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result is None:
            raise ValueError
        return result

    def get_example_pair(self, gpu_type):
        input_data1, input_data2, label_data= self['input_sentence1'], self['input_sentence2'], self['label']
        if (type(input_data1) is not MSRPInput) or (type(input_data2) is not MSRPInput) or (type(label_data) is not MSRPLabel):
            raise TypeError

        input_data1.align_sentence(self.input_align_length)
        input_data2.align_sentence(self.input_align_length)
        result = [input_data1.get_tensor(gpu_type), input_data2.get_tensor(gpu_type), label_data.get_tensor(gpu_type)]
        # if gpu_type:
        #     temp = []
        #     for data in result:
        #         temp.append(data.cuda())
        #     result = temp
        result = tuple(result)
        return result

    def remove_word_not_in_embedding_dictionary(self):
        input_data1, input_data2 = self['input_sentence1'], self['input_sentence2']
        input_data1.remove_word_not_in_embedding_dictionary()
        input_data2.remove_word_not_in_embedding_dictionary()






# class DataArray:
#     def __init__(self, data_element_list):
#         self.array = np.array(data_element_list, dtype=np.object)
#         #self.array = torch.from_numpy(self.array)


class MyDataSet(torch_data.dataset.Dataset):
    def __init__(self, data_array):
        if not isinstance(data_array[0], DataExample):
            raise TypeError

        super().__init__()
        self.data_array = data_array
        self.use_gpu_flag = False


    def __len__(self):
        return len(self.data_array)

    def __getitem__(self, item):
        result = self.data_array[item].get_example_pair(self.use_gpu_flag)
        for r in result:
            if not isinstance(r, torch.Tensor):
                raise TypeError
        return result

    def set_data_type_to_gpu(self):
        self.use_gpu_flag = True

    def set_data_type_to_cpu(self):
        self.use_gpu_flag = False

    def set_data_align_length(self, length = -1):
        for data in self.data_array:
            data.input_align_length = length

    def remove_word_not_in_embedding_dictionary(self):
        for data in self.data_array:
            data.remove_word_not_in_embedding_dictionary()


class DataManager:
    def __init__(self, original_file):
        self.file_operator = file_tool.FileOperator(original_file)
        self.original_data = self.file_operator.read(operation_func=lambda rows: rows, model = 'r')
        self.format_original_data()
        pass

    @abstractmethod
    def format_original_data(self):
        pass

    @abstractmethod
    def get_data_loader(self, batch_size):
        pass
        # return torch_data.dataloader.DataLoader(self.data_set, batch_size=batch_size)

    @abstractmethod
    def get_an_input(self, batch_size):
        raise RuntimeError('Do not implement this method')


class MSRPDataManager(DataManager):
    def __init__(self, original_file, name):
        self.data_set = None
        self.max_length = -1
        self.batch_size = 1
        self.number_of_word_not_in_dictionary = 0
        self.number_of_word = 0
        self.name = name
        self.sentence_pairs = []
        DataManager.__init__(self, original_file)

    def format_original_data(self):
        example_list = []
        self.original_data = self.original_data[1:]
        word_count = 0
        no_find_word_count = 0
        no_find_word_list = []
        for data in self.original_data:
            re_pattern = r'.+?[\t\n]'
            result=re.findall(re_pattern, data)
            label = MSRPLabel(result[0])
            input_sentence1 = MSRPInput(result[3])
            input_sentence2 = MSRPInput(result[4])
            self.sentence_pairs.append((input_sentence1,input_sentence2))
            example = MSRPDataExample(input_sentence1=input_sentence1, input_sentence2=input_sentence2, label=label)
            example_list.append(example)
            word_count += len(input_sentence1.sentence) + len(input_sentence2.sentence)
            no_find_word_count += len(input_sentence1.no_find_word_list) + len(input_sentence2.no_find_word_list)
            no_find_word_list.extend(input_sentence1.no_find_word_list)
            no_find_word_list.extend(input_sentence2.no_find_word_list)

        self.number_of_word = word_count
        self.number_of_word_not_in_dictionary = no_find_word_count
        log_str = "total word:{}, no_in_dictionary_word:{}\n".format(word_count, self.number_of_word_not_in_dictionary)
        log_str += str(no_find_word_list)
        file_tool.save_data(log_str, file_tool.PathManager.change_filename_by_append \
                            (file_tool.PathManager.word_no_in_dictionary_file, self.name), 'w')
        example_array = np.array(example_list, dtype=np.object)
        self.data_set = MyDataSet(example_array)
        self.get_max_length_of_sentence()

    def get_data_loader(self, drop_last, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        return torch_data.dataloader.DataLoader(self.data_set, batch_size=batch_size, drop_last=drop_last)

    def get_an_input(self):
        example = self.data_set[0]
        return example[0].unsqueeze(dim=0), example[1].unsqueeze(dim=0)


    def get_max_length_of_sentence(self):
        if self.max_length == -1:
            max_length = 0
            for pair in self.sentence_pairs:
                for sentence in pair:
                    if sentence.sentence_length > max_length:
                        max_length = sentence.sentence_length
            self.max_length = max_length
        return self.max_length

    def set_data_gpu_type(self, use_gpu):
        if use_gpu:
            self.data_set.set_data_type_to_gpu()
        else:
            self.data_set.set_data_type_to_cpu()

    def data_align(self):
        max_length = self.get_max_length_of_sentence()
        self.data_set.set_data_align_length(max_length)

    def remove_word_not_in_embedding_dictionary(self):
        self.data_set.remove_word_not_in_embedding_dictionary()

    def get_count_of_examples(self):
        return len(self.data_set)


MSRPC_Train_Manager_Single_Instance = None
MSRPC_Test_Manager_Single_Instance = None


def get_msrpc_manager(re_build = False):
    global MSRPC_Train_Manager_Single_Instance, MSRPC_Test_Manager_Single_Instance
    if MSRPC_Train_Manager_Single_Instance is None:
        if (not re_build) and os.path.isfile(file_tool.PathManager.msrpc_train_data_manager_path):
            train_manager = file_tool.load_data_pickle(file_tool.PathManager.msrpc_train_data_manager_path)
        else:
            train_manager = MSRPDataManager(file_tool.PathManager.msrpc_train_data_set_path, "train_manager")
            file_tool.save_data_pickle(train_manager, file_tool.PathManager.msrpc_train_data_manager_path)
        MSRPC_Train_Manager_Single_Instance = train_manager

    if MSRPC_Test_Manager_Single_Instance is None:

        if (not re_build) and os.path.isfile(file_tool.PathManager.msrpc_test_data_manager_path):
            test_manager = file_tool.load_data_pickle(file_tool.PathManager.msrpc_test_data_manager_path)
        else:
            test_manager = MSRPDataManager(file_tool.PathManager.msrpc_test_data_set_path, "test_manager")
            file_tool.save_data_pickle(test_manager, file_tool.PathManager.msrpc_test_data_manager_path)

        MSRPC_Test_Manager_Single_Instance = test_manager
    return MSRPC_Train_Manager_Single_Instance, MSRPC_Test_Manager_Single_Instance


def test():
    train_manager = MSRPDataManager(file_tool.PathManager.msrpc_train_data_set_path)
    loader = train_manager.get_data_loader(batch_size=1, drop_last=True)
    train_data = []
    for batch_input1s, batch_input2s, batch_labels in loader:
        train_data.append((batch_input1s, batch_input2s ,batch_labels))

    count = 10
    repeat = 0

    index = [random.randint(0, 299) for i in range(count)]
    batch1 = train_data[0][0][0][[0],index]
    batch2 = train_data[0][1][0][[0],index]

    for i in range(count):
        if batch1[i]==batch2[i]:
            repeat += 1
    print('重复率：{}'.format(repeat/count))

    get_msrpc_manager(re_build=True)

if __name__ == '__main__':
    test()