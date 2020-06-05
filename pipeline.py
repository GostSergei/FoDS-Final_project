

import torch


import datetime

import os
import copy
import json


class Trial():
    def __init__(self):
        return

    def initialization(self, trial_param, init_task_param):
        self.trial_param = copy.deepcopy(trial_param)
        self.init_task_param = copy.deepcopy(init_task_param)
        self.init_task()
        return

    def updating(self, trail_param):
        self.trial_param = copy.deepcopy(trail_param)
        self.init_task()
        return

    def init_task(self):
        self.update_init_task_param()
        task = self.trial_param['task_type']()
        task.initialisation(self.init_task_param)

        self.trial_param['task'] = task
        return

    def do_trial(self, n_epochs=-1, do_print=2, save=True):
        task = self.trial_param['task']

        task.do_train(n_epochs, do_print=do_print)
        value = task.do_estimate(n_epochs, do_print=do_print)
        self.trial_param['value'] = value

        self.inc_launch()
        if save:
            self.save_trail()

        return value

    def task(self):
        return self.trial_param['task']

    def value(self):
        return self.trial_param['value']

    def inc_launch(self):
        launch = self.trial_param['launch']
        launch += 1
        self.trial_param['launch'] = launch
        self.task().all_param['launch'] = launch
        return

    def update_init_task_param(self):
        hyper = self.trial_param['hyper']

        for key in hyper.keys():
            self.init_task_param[key] = hyper[key]

        self.init_task_param['rnn_type'] = self.trial_param['rnn_type']
        return

    def save_trail(self, file_name=''):
        self.save(file_name)
        self.save_log(file_name)
        return

    def save(self, file_name=''):
        ext = self.trial_param['save_ext']
        if file_name == '':
            file_name = self.gen_save_name() + ext

        save_dict = self.gen_save_dict()
        torch.save(save_dict, file_name)
        return

    def save_log(self, file_name=''):
        ext = self.trial_param['log_ext']
        if file_name == '':
            file_name = self.gen_save_name() + ext

        log_dict = self.get_main_trial_params()

        with open(file_name, 'w') as file:
            json.dump(log_dict, file, indent=4)
        return

    def gen_save_dict(self):
        trail_param = self.trial_param
        task = trail_param['task']
        trail_key = trail_param['trail_key']
        task_param = task.all_param

        save_dict = task.gen_save_dict(task_param)
        save_dict[trail_key] = trail_param
        return save_dict

    def load_trial(self, file_name):
        load_dict = torch.load(file_name)
        self.update_all(load_dict)
        return

    def update_all(self, load_dict):
        trail_key = self.trial_param['trail_key']

        trail_param = load_dict[trail_key]
        self.update_params(trail_param)

        task = self.trial_param['task']
        task.update_all(load_dict)
        return

    def update_params(self, params):
        for key in params.keys():
            self.trial_param[key] = params[key]
        self.updating(self.trial_param)
        return

    def gen_save_name(self):
        trail_param = self.trial_param
        # 'Trail_{launch}_{rnn_type}_{scale}_{learning_rate}_{n_layers}_{max_grad_norm}__{timemark}.tar'
        dir_path = trail_param['data_path']
        save_template = trail_param['save_template']

        rnn_type_name = trail_param['rnn_type'].name
        task_type = trail_param['task_type'].name
        launch = trail_param['launch']
        scale = trail_param['hyper']['scale']
        learning_rate = trail_param['hyper']['learning_rate']
        n_layers = trail_param['hyper']['n_layers']
        max_grad_norm = trail_param['hyper']['max_grad_norm']

        timemark = self.get_now_timemark()

        file_name = save_template.format(launch=launch, rnn_type_name=rnn_type_name, scale=scale,
                                         learning_rate=learning_rate, n_layers=n_layers, max_grad_norm=max_grad_norm,
                                         timemark=timemark, task_type=task_type)
        file_path = os.path.join(dir_path, file_name)

        return file_path

    def get_now_timemark(self):
        trail_param = self.trial_param
        timemark_format = trail_param['timemark_format']

        return (datetime.datetime.now().strftime(timemark_format))

    def get_main_trial_params(self):
        trail_param = self.trial_param
        keys = trail_param['main_trail_params']

        #         print(trail_param)
        main_trail_params = {}
        for key in keys:
            if key == 'rnn_type':
                main_trail_params[key] = trail_param[key].name
            elif key == 'task_type':
                main_trail_params[key] = trail_param[key].name
            elif key == 'value':
                main_trail_params[key] = round(trail_param[key], 5)
            else:
                main_trail_params[key] = trail_param[key]

        return main_trail_params


class TestRNNArch():
    def __init__(self):
        return

    def initialization(self, test_param, test_dict, init_trail_param, init_task_param,
                       hyper_list_policy='all_to_all', do_print=True):
        self.test_dict = copy.deepcopy(test_dict)
        self.test_param = copy.deepcopy(test_param)
        self.init_trail_param = copy.deepcopy(init_trail_param)
        self.init_task_param = copy.deepcopy(init_task_param)
        self.do_print = do_print

        self.last_results = []

        if hyper_list_policy == 'all_to_all':
            self.all_to_all_hyper_list()

        if self.do_print:
            print('TestRNNArch initializated good!')
            print("Lengh of hyper_list = ", len(self.hyper_list))
        return

    def do_test(self, rnn_arch, n_epochs, do_trail_print=1, do_save=False):
        hyper_list = self.hyper_list
        init_trail_param = self.init_trail_param
        rnn_type_key = self.test_param['rnn_type_key']
        tasks_list = self.test_param['tasks_list']
        tasks_key = self.test_param['tasks_key']
        test_print_template = self.test_param['test_print_template']
        posttest_print_template = self.test_param['posttest_print_template']
        init_task_param = self.init_task_param

        param_dict = {rnn_type_key: rnn_arch}

        final_result_list = []
        for task in tasks_list:
            param_dict[tasks_key] = task
            result_list = []
            if self.do_print:
                print('\nStart task {}:'.format(task.name))
            for i in range(len(hyper_list)):
                trail_param = init_trail_param
                hyper = hyper_list[i]
                self.set_dict(trail_param, hyper)
                self.set_dict(trail_param, param_dict)

                trial = Trial()
                trial.initialization(trail_param, init_task_param)
                trial.do_trial(n_epochs, do_trail_print, do_save)
                value = trial.value()
                result_list.append(trial.get_main_trial_params())
                if self.do_print:
                    # 'no {} | value= {:.2f}% | params: sc={}, lr={}, n_l={}, m_g={}|'
                    print(test_print_template.format(i, value * 100, hyper['scale'], hyper['learning_rate'],
                                                     hyper['n_layers'], hyper['max_grad_norm']))

            best_result = self.find_best(result_list)
            if self.do_print:
                print('Task {} ended:'.format(task.name))
                print(posttest_print_template.format(best_result['value'] * 100, best_result['hyper']['scale'],
                                                     best_result['hyper']['learning_rate'],
                                                     best_result['hyper']['n_layers'],
                                                     best_result['hyper']['max_grad_norm']))
            final_result_list.append(best_result)

        self.last_results = final_result_list
        return final_result_list

    def get_results_hat(self, results=None):
        tasks_key = self.test_param['tasks_key']
        tasks_list = self.test_param['tasks_list']
        out_list = [self.test_param['results_start_hat']]
        if results is None:
            out_list += [tasks_list[i].name for i in  range(len(tasks_list))]
        else:
            out_list += [ results[i][tasks_key] for i in  range(len(results))]

        out = ''
        sep = self.test_param['results_sep']
        results_cell = self.test_param['results_cell']
        for i in range(len(out_list)):
            cell = out_list[i]
            n = results_cell - len(cell)
            cell += sep*n
            out += cell
        return out

    def get_results(self, results=None):
        rnn_type_key = self.test_param['rnn_type_key']
        value_key = self.test_param['value_key']
        if results is None:
            results = self.last_results

        sep = self.test_param['results_sep']
        results_cell = self.test_param['results_cell']

        cell = results[0][rnn_type_key]
        out = cell + sep * (results_cell - len(cell))
        for i in range(len(results)):
            cell = str(round(results[i][value_key], 5))
            out += cell + sep * (results_cell - len(cell))

        return out

    def set_dict(self, target, param):
        for key in param.keys():
            target[key] = param[key]

    def find_best(self, result_list):
        value_key = self.test_param['value_key']
        best_result = result_list[0]

        for i in range(1, len(result_list)):
            if result_list[i][value_key] > best_result[value_key]:
                best_result = result_list[i]
        return best_result

    def all_to_all_hyper_list(self):
        test_dict = self.test_dict
        hyper_key = self.test_param['hyper_key']
        test_dict
        hyper_list = []

        keys = list(test_dict.keys())
        for i in range(len(test_dict[keys[0]]['list'])):
            for ii in range(len(test_dict[keys[1]]['list'])):
                for iii in range(len(test_dict[keys[2]]['list'])):
                    for iiii in range(len(test_dict[keys[3]]['list'])):
                        d = {
                            test_dict[keys[0]]['key']: test_dict[keys[0]]['list'][i],
                            test_dict[keys[1]]['key']: test_dict[keys[1]]['list'][ii],
                            test_dict[keys[2]]['key']: test_dict[keys[2]]['list'][iii],
                            test_dict[keys[3]]['key']: test_dict[keys[3]]['list'][iiii],
                        }
                        #                         hiper = {hyper_key:d}
                        hyper_list.append(d)
        self.hyper_list = hyper_list
        return hyper_list



