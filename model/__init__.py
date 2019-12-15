ws_endless_flag = True
input_align = True
show_run_time = False
log_model_data_flag =True

word_vector_dim = 300
learn_rate = 0.01
sgd_momentum = 0.4
batch_size = 64
filter_number_a = 525
filter_number_b = 25
wss_a = [1, 2, 3]
wss_b = [1, 2, 3]
poolings_a = ['max', 'min', 'mean']
poolings_b = ['max','min']
block_model_names = ['BlockA', 'BlockB']
compare_unit_names = ['cos', 'l2', 'l1']
compare_model_names= ['Horizontal', 'VerticalForBlockA', 'VerticalForBlockB']
output_size_of_compare_model_dict = {
    'Horizontal': len(poolings_a)*filter_number_a*len(compare_unit_names),
    'VerticalForBlockA': len(poolings_a)*(len(wss_a)**2)*len(compare_unit_names),
    'VerticalForBlockB': len(poolings_b)*len(wss_b)*filter_number_b*len(compare_unit_names)
}
hidden_cell_number = 250


sub_model_select_dict = [
    {
        'block_model': 'BlockB',
        'compare_model': 'VerticalForBlockB'
    },
    {
        'block_model': 'BlockA',
        'compare_model': 'VerticalForBlockA'
    },
    {
        'block_model': 'BlockA',
        'compare_model': 'Horizontal'
    }
]




