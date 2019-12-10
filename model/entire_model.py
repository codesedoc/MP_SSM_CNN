import model.sentence_model as sentence_model
import model.similarity_measure_model as similarity_measure_model
import model.full_connect_model as full_connect_model
import torch
import model as model_init
import time

class EntireModel(torch.nn.Module):
    def __init__(self,number,word_vector_dim, wss, compare_unit_names):
        super().__init__()
        poolings = [sentence_model.pooling_model_dict['max'],
                    sentence_model.pooling_model_dict['min'],
                    sentence_model.pooling_model_dict['mean']]
        horizontal_compare = True
        self.sentence_model = sentence_model.SentenceModel(block_type='BlockA', wss=wss, poolings=poolings, number=number,
                                                           word_vector_dim=word_vector_dim)
        self.similarity_measure_model = similarity_measure_model.SimilarityMeasureModel(block_type='BlockA',
                                                            compare_unit_names=compare_unit_names, horizontal_compare=horizontal_compare)
        before_input = len(compare_unit_names)
        input_layer = len(poolings)*number*before_input
        #
        self.full_connect_model = full_connect_model.FullConnectModel((input_layer, 250, 2))

    def forward(self, input_data1, input_data2):
        if model_init.show_run_time:
            start = time.time()
            result1 = self.sentence_model(input_data1)
            result2 = self.sentence_model(input_data2)
            end = time.time()
            print("sentence_model_run_time:{}".format(end - start))
        else:
            result1 = self.sentence_model(input_data1)
            result2 = self.sentence_model(input_data2)

        if model_init.show_run_time:
            start = time.time()
            result = self.similarity_measure_model(result1,result2)
            end = time.time()
            print("similarity_measure_model_run_time:{}".format(end - start))
        else:
            result = self.similarity_measure_model(result1,result2)

        if model_init.show_run_time:
            start = time.time()
            result = self.full_connect_model(result)
            end = time.time()
            print("full_connect_model_run_time:{}".format(end - start))
        else:
            result = self.full_connect_model(result)


        return result

