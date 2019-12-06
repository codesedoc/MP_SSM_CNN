import utils.data_tool as data_tool


ws_endless_flag = True
input_align = True
num_filter_a = 525
# train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
# ws_max = max(train_manager.get_max_length_of_sentence(), test_manager.get_max_length_of_sentence())


class Preprocessor:
    def __init__(self):
        pass

    def pre_process(self, data_manager, batch_size, use_gpu, data_align, remove_error_word_vector):
        data_manager.set_data_gpu_type(use_gpu)
        data_manager.batch_size = batch_size
        if data_align:
            data_manager.data_align()
        if remove_error_word_vector:
            data_manager.remove_word_not_in_embedding_dictionary()
        return data_manager
