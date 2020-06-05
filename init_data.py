import os
import torch.nn as nn
from rnn_cells import EncoderCell, DecoderCell, MyLinerRNN_cell, MyRNN_cell, MyLSTM_cell, MyGRU_cell
from tasks import ArithmeticTask, XMLTask, FreeTask, PTBTask

project_path = os.getcwd()
work_dir = 'Work'
data_dir = 'Data'

work_path = os.path.join(project_path, work_dir)
data_path = os.path.join(project_path, data_dir)

launch = 1



test_dict = {
    'scale': {
        'list': [1],
        'key': 'scale',
    },
    'learning_rate': {
        'list': [1, 30],
        'key': 'learning_rate',
    },
    'n_layers': {
        'list': [1],
        'key': 'n_layers',
    },
    'max_grad_norm': {
        'list': [1],
        'key': 'max_grad_norm',
    },
}

# test_dict = {
#     'scale': {
#         'list':[0.3, 0.7, 1, 1.4, 2, 2.8],
#         'key': 'scale',
#     },
#     'learning_rate': {
#         'list':[0.1, 0.2, 0.3, 0.5, 1, 2, 5],
#         'key': 'learning_rate',
#     },
#     'n_layers': {
#         'list':[1, 2, 3, 4],
#         'key': 'n_layers',
#     },
#     'max_grad_norm': {
#         'list':[1, 2.5, 5, 10, 20],
#         'key': 'max_grad_norm',
#     },
# }

test_param = {
    # 'tasks_list':[ArithmeticTask, FreeTask],
    'tasks_list': [ArithmeticTask, XMLTask, PTBTask],
    'tasks_key': 'task_type',
    'hyper_key': 'hyper',
    'rnn_type_key': 'rnn_type',
    'value_key': 'value',
    'work_path': work_path,
    'data_path': data_path,
    'test_print_template': 'no{} | value= {:.2f}% | params: sc={}, lr={}, n_l={}, m_g={} |',
    'posttest_print_template': 'Best trial | value= {:.2f}% | params: sc={}, lr={}, n_l={}, m_g={} |',
    'results_start_hat': "Architacture",
    'results_sep': ' ',
    'results_cell': 15,
}

trial_param = {
    'rnn_type': MyGRU_cell,
    'hyper': {
        'scale': 1,
        'learning_rate': 0.005,
        'n_layers': 1,
        'max_grad_norm': 10
    },
    'value': -1,
    'task_type': XMLTask,  # ArithmeticTask,
    'task': None,
    #     'task_param': all_param,''
    'trail_key': 'trail_param',
    'work_path': work_path,
    'data_path': data_path,
    'launch': 0,
    'save_template': 'Trail_{launch}_{rnn_type_name}_{task_type}_{scale}_{learning_rate}_' +
                     '{n_layers}_{max_grad_norm}__{timemark}',
    'save_ext': '.tar',
    'log_ext': '.json',
    'timemark_format': '%d_%H_%M',
    'main_trail_params': ['rnn_type', 'hyper', 'value', 'task_type', 'launch']

}

# parameters
task_param = {}
# For project
task_param['launch'] = launch
task_param['work_path'] = work_path
task_param['data_path'] = data_path

# RNN param
task_param['decoder_type'] = DecoderCell
task_param['encoder_type'] = EncoderCell
task_param['n_layers'] = 1
task_param['n_hidden'] = 128
task_param['rnn_type'] = MyGRU_cell
# all_param['forget_baise_init'] = 1
# all_param['init_baises'] = False
task_param['batch_size'] = 1
task_param['learning_rate'] = 0.1
# all_param['softmax_fun'] =  nn.Softmax
task_param['criterion'] = nn.NLLLoss()

# for training
task_param['max_X_lenght'] = 15
task_param['max_Y_lenght'] = 15
task_param['device_name'] = "cpu"
task_param['optimizer_name'] = 'SGD'
task_param['max_grad_norm'] = 10
task_param['scale'] = 1

# do traning
task_param['n_epochs'] = 100000
task_param['print_every'] = 5000
task_param['plot_every'] = 1000
task_param['refresh_frequency'] = 1  # number of refreshing in a train

# For estimating
task_param['n_estimates'] = 10000
task_param['est_print_every'] = 1000
task_param['est_batch_size'] = 1

# for logging

task_param['timemark_format'] = "%d_%m_%Y__%H_%M_%S"
task_param['timemark_print_format'] = "Date: %Y.%m.%d time: %H:%M"
task_param['refresh_template'] = "refresh_{rnn_name}_{timemark}.pt"
task_param['save_template'] = "save_{rnn_name}_{launch}_{n_train}__{timemark}.tar"
task_param['log_template'] = 'Log__{rnn_name}_{launch}_{n_train}__{timemark}.json'

task_param['save_dict_keys'] = ['model_state', 'optimizer_state', 'params']
task_param['model_state_dict'] = {'decoder': None, 'encoder': None}
task_param['optimize_state_dict'] = {'decod_optimizer': None, 'encod_optimizer': None}
task_param['save_model_state_i'] = 0
task_param['save_optimizer_state_i'] = 1
task_param['save_params_i'] = 2
task_param['save_main_params_list'] = ['rnn_type', 'learning_rate', 'current_epochs', 'current_value',
                                      'current_loss', 'batch_size', 'n_hidden', 'total_train_time',
                                      'criterion', 'optimizer_name', 'device_name']
task_param['task_params_list'] = ['current_epochs', 'current_value', 'current_loss', 'total_train_time',
                                 'last_train_time', 'all_train_times', 'all_epochs', 'all_values',
                                 'all_losses']
# for tasks
task_param['current_epochs'] = 0
task_param['current_value'] = 0
task_param['current_loss'] = 0
task_param['last_train_time'] = 0
task_param['total_train_time'] = 0

task_param['all_train_times'] = []
task_param['all_epochs'] = []
task_param['all_values'] = []
task_param['all_losses'] = []

