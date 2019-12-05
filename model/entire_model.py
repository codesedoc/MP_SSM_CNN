import model.sentence_model as sentence_model
import model.similarity_measure_model as similarity_measure_model
import model.full_connect_model as full_connect_model
import torch
import model

class EntireModel(torch.nn.Module):
    def __init__(self,number,word_vector_dim):
        super().__init__()
        wss = [1, 2, 3]
        poolings =[sentence_model.pooling_model_dict['max'],
                   sentence_model.pooling_model_dict['min'],
                   sentence_model.pooling_model_dict['mean']]
        compare_units = ['cos','l2','l1']

        horizontal_compare = True
        self.sentence_model = sentence_model.SentenceModel(block_type='BlockA', wss=wss, poolings=poolings, number=number,
                                                           word_vector_dim=word_vector_dim)
        self.similarity_measure_model = similarity_measure_model.SimilarityMeasureModel(block_type='BlockA',
                                                            compare_units=compare_units, horizontal_compare=horizontal_compare)
        before_input = len(compare_units)
        input_layer = len(poolings)*number*before_input
        #
        self.full_connect_model = full_connect_model.FullConnectModel((input_layer, 250, 2))

    def forward(self, input_data1, input_data2):
        result1=self.sentence_model(input_data1)
        result2 = self.sentence_model(input_data2)
        result=self.similarity_measure_model(result1,result2)
        result = result.reshape(result.size()[0], -1)
        result=self.full_connect_model(result)
        return result

