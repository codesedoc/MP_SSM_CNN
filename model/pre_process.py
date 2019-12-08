import utils.data_tool as data_tool
import utils.file_tool as file_tool
# train_manager, test_manager = data_tool.get_msrpc_manager(re_build=False)
# ws_max = max(train_manager.get_max_length_of_sentence(), test_manager.get_max_length_of_sentence())


class Preprocessor:
    def __init__(self):
        pass

    def pre_process(self, data_manager, batch_size, use_gpu, data_align, remove_error_word_vector, rebuild = False):
        file_path = file_tool.PathManager.append_filename_to_dir_path(file_tool.PathManager.msrpc_path,
                                                                      data_manager.name + '_preprocessed', extent='pkl')
        if rebuild:
            data_manager.set_data_gpu_type(use_gpu)
            data_manager.batch_size = batch_size
            if data_align:
                data_manager.data_align()
            if remove_error_word_vector:
                data_manager.remove_word_not_in_embedding_dictionary()
            file_tool.save_data_pickle(data_manager, file_path)
        else:
            data_manager = file_tool.load_data_pickle(file_path)
        return data_manager
