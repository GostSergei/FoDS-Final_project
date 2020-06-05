import glob
import unicodedata
import string
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn

import datetime
from abc import ABC, abstractmethod
import os
import copy
import json


from collections import defaultdict
from torchnlp.datasets import penn_treebank_dataset
from torchnlp.encoders import LabelEncoder
from torchnlp.nn import LockedDropout
from torchnlp.nn import WeightDrop
from torchnlp.samplers import BPTTBatchSampler

from PTB_module import repackage_hidden, RNNModel, SplitCrossEntropyLoss

class AbstractTask(ABC):
    def __init__(self):
        return

    @abstractmethod
    def initialisation(self, all_parameters):
        # Parameters
        self.all_param = copy.deepcopy(all_parameters)
        all_param = self.all_param
        # launch
        # device
        device_name = all_param['device_name']
        # print(device_name)
        if device_name == "cuda":
            if not torch.cuda.is_available():
                device_name = "cpu"
                print("GPU not available")
        device = torch.device(device_name)

        all_param['device_name'] = device_name
        all_param['device'] = device

        #         if device_name == "cuda":
        #             print('Device -> GPU')
        #         elif device_name == "cpu":
        #             print('Device -> CPU')

        # random
        random.seed()

        # RNN param

        # learning param
        optimizer_name = all_param['optimizer_name']
        #         print('Optimizer ->', optimizer_name )

        # train param

        # task history
        # for tasks

        #         all_param['last_train_time'] = 0
        #         all_param['total_train_time'] = 0
        #         all_param['all_train_times'] = [0]

        # others

        return

    @abstractmethod
    def do_train(self):
        return

    @abstractmethod
    def do_estimate(self):
        return

    @abstractmethod
    def get_rnn(self):
        return

    @abstractmethod
    def get_optimizer(self):
        return

    def plot_total_train(self):
        plt.figure()
        plt.plot(self.all_param['all_losses'])

    def log_train_results(self, plus_epochs, plus_loss_list, start_time, stop_time):
        all_param = self.all_param
        get_total_train_time_ = self.get_total_train_time_

        all_param['current_epochs'] += plus_epochs
        all_param['current_loss'] = plus_loss_list[-1]
        all_param['all_epochs'] += [plus_epochs]
        all_param['all_losses'] += plus_loss_list

        last_train_time = stop_time - start_time
        all_param['last_train_time'] = last_train_time
        all_param['total_train_time'] = get_total_train_time_() + last_train_time
        all_param['all_train_times'] = [last_train_time]
        return

    def get_total_train_time_(self):
        all_param = self.all_param
        all_train_times = all_param['all_train_times']
        total_train_time = 0
        for t in all_train_times:
            total_train_time += t
        return total_train_time

    def log_estimate_results(self, value):
        all_param = self.all_param
        all_param['current_value'] = value
        all_param['all_values'] += [value]
        return

    def get_main_params(self):
        all_param = self.all_param
        get_params_dict = AbstractTask.get_params_dict

        return get_params_dict(all_param['save_main_params_list'], all_param)

    def get_task_params(self):
        all_param = self.all_param
        get_params_dict = AbstractTask.get_params_dict

        return get_params_dict(all_param['task_params_list'], all_param)

    def show_main_params(self, print_=True):
        all_param = self.all_param
        get_params_str_dict = self.get_params_str_dict

        main_params = get_params_str_dict(all_param['save_main_params_list'])
        if print_: print(main_params)
        return main_params

    def load_params(self, file_name):
        loaded_dict = torch.load(file_name)
        self.update_all(loaded_dict)

    def update_all(self, save_dict):
        all_param = self.all_param
        get_rnn = self.get_rnn
        get_optimizer = self.get_optimizer
        update_params = self.update_params

        keys = all_param['save_dict_keys']
        model_state_key = keys[all_param['save_model_state_i']]
        optimizer_state_key = keys[all_param['save_optimizer_state_i']]
        params_key = keys[all_param['save_params_i']]

        update_params(save_dict[params_key])

        encoder, decoder = get_rnn()
        encod_optimizer, decod_optimizer = get_optimizer()

        encoder.load_state_dict(save_dict[model_state_key]['encoder'])
        decoder.load_state_dict(save_dict[model_state_key]['decoder'])

        encod_optimizer.load_state_dict(save_dict[optimizer_state_key]['encod_optimizer'])
        decod_optimizer.load_state_dict(save_dict[optimizer_state_key]['decod_optimizer'])

        decoder.train()
        encoder.train()

    def update_params(self, params):
        all_param = self.all_param
        initialisation = self.initialisation

        for key in params.keys():
            all_param[key] = params[key]
        initialisation(all_param)
        return

    def save_log(self, file_name=''):
        get_log_file_name = self.get_log_file_name
        show_main_params = self.show_main_params

        if file_name == '':
            file_name = get_log_file_name()

        params = show_main_params(False)

        with open(file_name, 'w') as file:
            json.dump(params, file, indent=4)

        return

    def save_all_params(self):
        all_param = self.all_param
        save_params = self.save_params

        pref = 'All_'
        params = all_param
        save_params(params, pref)

    def save_main_params(self):
        all_param = self.all_param
        save_params = self.save_params
        get_params_dict = AbstractTask.get_params_dict

        pref = 'main_'
        params = get_params_dict(all_param['save_main_params_list'], all_param)
        save_params(params, pref)

    def save_params(self, params, pref=''):
        get_save_file_name = self.get_save_file_name
        gen_save_dict = self.gen_save_dict

        save_file_name = get_save_file_name(pref)
        save_dict = gen_save_dict(params)

        torch.save(save_dict, save_file_name)
        return

    def gen_save_dict(self, params):
        all_param = self.all_param
        get_rnn = self.get_rnn
        get_optimizer = self.get_optimizer

        dir_path = all_param['data_path']
        keys = all_param['save_dict_keys']
        model_state_key = keys[all_param['save_model_state_i']]
        optimizer_state_key = keys[all_param['save_optimizer_state_i']]
        params_key = keys[all_param['save_params_i']]

        encoder, decoder = get_rnn()
        model_state = {'decoder': decoder.state_dict(), 'encoder': encoder.state_dict()}
        encod_optimizer, decod_optimizer = get_optimizer()
        optimizer_state = {'encod_optimizer': encod_optimizer.state_dict(),
                           'decod_optimizer': decod_optimizer.state_dict()}

        save_dict = {model_state_key: model_state, optimizer_state_key: optimizer_state,
                     params_key: params}
        return save_dict

    def get_params_dict(save_params_list, all_param):
        params = {}
        for key in save_params_list:
            params[key] = all_param[key]
        return params

    def get_params_str_dict(self, save_params_list):
        all_param = self.all_param
        rnn_type = all_param['rnn_type']
        get_rnn = self.get_rnn

        params = {}
        for key in save_params_list:
            if all_param[key] == rnn_type:
                params[key] = rnn_type.name
            else:
                params[key] = str(all_param[key])
        return params

    def refresh(self):
        all_param = self.all_param
        get_rnn = self.get_rnn
        get_optimizer = self.get_optimizer
        get_refresh_file_name = self.get_refresh_file_name

        dir_path = all_param['work_path']
        keys = all_param['save_dict_keys']
        model_state_key = keys[all_param['save_model_state_i']]
        optimizer_state_key = keys[all_param['save_optimizer_state_i']]

        # "refresh_{rnn_name}_{timemark}.pt"
        decoder, encoder = get_rnn()
        model_state = {'decoder': decoder.state_dict(), 'encoder': encoder.state_dict()}
        decod_optimizer, encod_optimizer = get_optimizer()
        optimizer_state = {'encod_optimizer': encod_optimizer.state_dict(),
                           'decod_optimizer': decod_optimizer.state_dict()}
        refresh_file_name = get_refresh_file_name()

        refresh_dict = {model_state_key: model_state, optimizer_state_key: optimizer_state}
        torch.save(refresh_dict, refresh_file_name)

        refresh_dict = torch.load(refresh_file_name)

        encoder.load_state_dict(refresh_dict[model_state_key]['encoder'])
        decoder.load_state_dict(refresh_dict[model_state_key]['decoder'])
        encod_optimizer.load_state_dict(refresh_dict[optimizer_state_key]['encod_optimizer'])
        decod_optimizer.load_state_dict(refresh_dict[optimizer_state_key]['decod_optimizer'])
        decoder.train()
        encoder.train()
        return

    def get_refresh_file_name(self):
        all_param = self.all_param
        get_now_timemark = self.get_now_timemark
        rnn_type = all_param['rnn_type']
        refresh_template = all_param['refresh_template']
        dir_path = all_param['work_path']

        refresh_file_name = refresh_template.format(rnn_name=rnn_type.name,
                                                    timemark=get_now_timemark())
        refresh_file_name = os.path.join(dir_path, refresh_file_name)
        return refresh_file_name

    def get_log_file_name(self, pref=''):
        all_param = self.all_param
        get_now_timemark = self.get_now_timemark
        rnn_type = all_param['rnn_type']
        all_epochs = all_param['all_epochs']
        log_template = all_param['log_template']
        dir_path = all_param['data_path']
        launch = all_param['launch']

        n_train = len(all_epochs)
        file_name = log_template.format(rnn_name=rnn_type.name,
                                        timemark=get_now_timemark(),
                                        launch=launch,
                                        n_train=n_train)

        file_name = os.path.join(dir_path, pref + file_name)
        return file_name

    def get_save_file_name(self, pref=''):
        all_param = self.all_param
        get_now_timemark = self.get_now_timemark
        rnn_type = all_param['rnn_type']
        all_epochs = all_param['all_epochs']
        save_template = all_param['save_template']
        dir_path = all_param['data_path']
        launch = all_param['launch']

        n_train = len(all_epochs)
        save_file_name = save_template.format(rnn_name=rnn_type.name,
                                              timemark=get_now_timemark(),
                                              launch=launch,
                                              n_train=n_train)

        save_file_name = os.path.join(dir_path, pref + save_file_name)
        return save_file_name

    def get_now_timemark(self):
        all_param = self.all_param
        timemark_format = all_param['timemark_format']

        return (datetime.datetime.now().strftime(timemark_format))

    def symbol_to_tensor(symbol, all_symbols):
        symbol_i = all_symbols.index(symbol)
        l = len(all_symbols)
        tensor = torch.zeros(l)
        tensor[symbol_i] = 1
        return tensor

    def one_hot_encoding(line, all_symbols, max_length=-1):
        l = len(all_symbols)
        n = 1  # batch_size
        batch_i = 0  # batch index

        if max_length > 0:
            m = max_length
        else:
            m = len(line)

        tensor = torch.zeros(m, n, l)

        for i, s in enumerate(line):
            s_i = all_symbols.index(s)
            tensor[i][batch_i][s_i] = 1

        return tensor

    def one_hot_decoding(tensor, all_symbols, batch_first=False):
        line_list = []
        if batch_first:
            batch_dim = 0
            s_dim = 1
        else:
            batch_dim = 1
            s_dim = 0

        l = len(all_symbols)
        for j in range(tensor.size(batch_dim)):
            line = ''
            for i in range(tensor.size(s_dim)):
                t_of_i = tensor[i][j].nonzero()
                s = ''
                if t_of_i.size(0) == 0:
                    break
                elif t_of_i.size(0) == 1:
                    s = all_symbols[t_of_i]
                else:
                    print("Error! Not one-hot vector!!!")
                    return
                line += s

            line_list.append(line)
        return line_list

    def tensor_to_one_hot(tensor, batch_first=False):
        k = 1
        dim = 2
        if batch_first:
            batch_dim = 0
            s_dim = 1
        else:
            batch_dim = 1
            s_dim = 0

        m = tensor.size(s_dim)
        n = tensor.size(batch_dim)
        top_v, top_i = torch.topk(tensor, k, dim=dim)

        one_hot_tensor = torch.zeros(tensor.size())
        for i in range(m):
            for j in range(n):
                if tensor[i][j].nonzero().size(0) != 0:
                    one_hot_tensor[i][j][top_i[i][j]] = 1

        return (one_hot_tensor)

    def create_vocab(all_symbols, EOS='<eos>', SOS='<sos>'):
        vocab = {}
        for i, symbol in enumerate(all_symbols):
            vocab[symbol] = i

        keys = vocab.keys()
        if SOS not in keys:
            vocab[SOS] = len(keys)

        keys = vocab.keys()
        if EOS not in keys:
            vocab[EOS] = len(keys)

        EOS_ix = vocab[EOS]
        SOS_ix = vocab[SOS]

        return vocab, EOS_ix, SOS_ix

    def output_to_tensor_line(output):
        topv, topi = output.topk(1)
        tensor_line = topi.squeeze(2).detach()  # detach from history as input
        return tensor_line

    def indexes_from_line(line, vocab):
        return [vocab[s] for s in line]

    def tensor_line_from_line(line, vocab, EOS_token, max_length=-1):
        indexes_from_line = AbstractTask.indexes_from_line

        indexes = indexes_from_line(line, vocab)
        n = len(indexes)
        if (max_length < 0):
            max_length = n + 1
        for i in range(n, max_length):
            indexes.append(EOS_token)

        return torch.tensor(indexes[:max_length], dtype=torch.long)

    def clipping_tensor(tensor, save_last=True):
        end_part = tensor[-1]
        n = 0
        for i in range(tensor.size(0)):
            if torch.equal(tensor[i], end_part):
                n = i
                break

        if save_last:
            if n + 1 < tensor.size(0):
                clipped_tensor = tensor.narrow(0, 0, n + 1)
            else:
                clipped_tensor = tensor
        else:
            clipped_tensor = tensor.narrow(0, 0, n)

        return clipped_tensor

    def lines_from_tensor_line(tensor, vocab, EOS_token, out_EOS=True, batch_first=False):
        if batch_first:
            batch_dim = 0
            s_dim = 1
        else:
            batch_dim = 1
            s_dim = 0

        vocab_list = [key for key in vocab.keys()]
        batch_size = tensor.size(batch_dim)
        step_size = tensor.size(s_dim)
        lines = ['' for i in range(batch_size)]
        for j in range(batch_size):
            i = 0
            while i < step_size:
                if tensor[i][j].item() == EOS_token:
                    if out_EOS:
                        lines[j] += vocab_list[tensor[i][j].item()]
                    break
                else:
                    lines[j] += vocab_list[tensor[i][j].item()]

                i += 1

        return lines

    def compare_lines(line1, line2):
        minlen = min(len(line1), len(line2))
        maxlen = max(len(line1), len(line2))
        value = 0
        for i in range(minlen):
            if line1[i] == line2[i]:
                value += 1
        value = value / maxlen
        return value

    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

class ArithmeticTask(AbstractTask):
    name = 'Arithmetic'

    def __init__(self):
        super().__init__()
        return

    def get_rnn(self):
        return (self.all_param['encoder'], self.all_param['decoder'])

    def get_optimizer(self):
        return (self.all_param['encod_optimizer'], self.all_param['decod_optimizer'])

    def initialisation(self, all_parameters):
        # parameters
        super().initialisation(all_parameters)
        create_vocab = AbstractTask.create_vocab
        all_param = self.all_param
        # task param
        self.distractor_symbols = string.ascii_lowercase
        self.clear_symbols = '0123456789' + '+-=.'
        self.all_symbols = self.clear_symbols + self.distractor_symbols

        self.vocab, self.EOS_token, self.SOS_token = create_vocab(self.all_symbols)
        self.vocab_decod, self.EOS_token_, self.SOS_token_ = create_vocab(self.clear_symbols)

        vocab = self.vocab
        vocab_decod = self.vocab_decod
        self.EOS_token_encod = vocab['=']

        self.SOS_token_decod = vocab_decod['<sos>']
        self.EOS_token_decod = vocab_decod['.']
        # RNN param
        vocab_size = len(self.vocab)
        vocab_decod_size = len(self.vocab_decod)

        hidden_size = all_param['n_hidden']
        device = all_param['device']
        rnn_type = all_param['rnn_type']
        encoder_type = all_param['encoder_type']
        decoder_type = all_param['decoder_type']
        n_layers = all_param['n_layers']
        scale = all_param['scale']

        decoder = decoder_type(vocab_decod_size, hidden_size, rnn_type, scale, n_layers)
        encoder = encoder_type(vocab_size, hidden_size, rnn_type, scale, n_layers)
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        all_param['decoder'] = decoder
        all_param['encoder'] = encoder

        # training param
        criterion = all_param['criterion']
        #         print('Criterion ->', criterion)

        optimizer_name = all_param['optimizer_name']
        learning_rate = all_param['learning_rate']
        if optimizer_name == 'SGD':
            decod_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
            encod_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)

        all_param['decod_optimizer'] = decod_optimizer
        all_param['encod_optimizer'] = encod_optimizer

        # estimating

    def do_train(self, n_epochs=-1, do_print=2):
        #         vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod
        distractor_symbols = self.distractor_symbols

        all_param = self.all_param

        decoder = all_param['decoder']
        encoder = all_param['encoder']
        batch_size = all_param['batch_size']
        max_X_lenght = all_param['max_X_lenght']
        max_Y_lenght = all_param['max_Y_lenght']
        timemark_print_format = all_param['timemark_print_format']
        print_every = all_param['print_every']
        plot_every = all_param['plot_every']
        refresh_frequency = all_param['refresh_frequency']

        train = self.train
        refresh = self.refresh
        gen_random_batch = self.gen_random_batch
        output_to_tensor_line = AbstractTask.output_to_tensor_line
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        time_since = AbstractTask.time_since

        if n_epochs < 0:
            n_epochs = all_param['n_epochs']

        current_loss = 0
        all_losses = []

        guess_i = 0

        Y_srt_sample = ''
        start = time.time()

        for epoch in range(1, n_epochs + 1):
            # prepare traning_batch and category_batch
            Y_batch, X_batch = gen_random_batch(batch_size, distractor_symbols)
            # training
            output, loss = train(Y_batch, X_batch, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)
            Y_srt_sample = lines_from_tensor_line(Y_batch, vocab_decod, EOS_decod)[guess_i]
            del Y_batch
            del X_batch

            current_loss += loss.item()
            if do_print >= 2:
                if epoch % print_every == 0:
                    Y_hat = output_to_tensor_line(output)
                    Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)
                    guess = Y_hat_str[guess_i]
                    correct = '✓' if guess == Y_srt_sample else '✗ (%s)' % Y_srt_sample

                    print('%d %d%% (%s) %.4f / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start),
                                                         loss, guess, correct))

            if epoch % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

            if epoch == n_epochs // refresh_frequency:
                refresh()
                if do_print >= 2:
                    print('Refreshed!')

        stop = time.time()
        learn_time = time_since(start)
        learn_datetime = datetime.datetime.now()
        data_str = learn_datetime.strftime(timemark_print_format)

        if len(all_losses) == 0:
            all_losses += [loss]
        self.log_train_results(n_epochs, all_losses, start, stop)
        if do_print >= 1:
            print("Learning ended. Learn datetime: " + data_str, 'Learn time: ' + str(learn_time))
        if do_print >= 2:
            plt.figure()
            plt.plot(all_losses)
        return

    def do_estimate(self, n_estimates=-1, do_print=2):
        #         vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod
        distractor_symbols = self.distractor_symbols

        all_param = self.all_param
        decoder = all_param['decoder']
        encoder = all_param['encoder']
        #         batch_size = all_param['batch_size']
        max_X_lenght = all_param['max_X_lenght']
        max_Y_lenght = all_param['max_Y_lenght']
        timemark_print_format = all_param['timemark_print_format']
        print_every = all_param['est_print_every']

        evaluate = self.evaluate
        refresh = self.refresh
        gen_random_batch = self.gen_random_batch
        estimate_performance = ArithmeticTask.estimate_performance
        output_to_tensor_line = AbstractTask.output_to_tensor_line
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        time_since = AbstractTask.time_since

        if n_estimates < 0:
            n_estimates = all_param['n_estimates']

        n_epochs = n_estimates

        est_batch_size = 1
        guess_i = 0

        start = time.time()

        with torch.no_grad():
            mean_value = 0
            for epoch in range(1, n_estimates + 1):
                # generation
                Y, X = gen_random_batch(est_batch_size, distractor_symbols)
                Y_str = lines_from_tensor_line(Y, vocab_decod, EOS_decod)

                # evaluating
                Y_hat, output = evaluate(X, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)

                # estimating
                Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)
                value = estimate_performance(Y_str, Y_hat_str)
                mean_value += value
                if do_print >= 2:
                    if epoch % print_every == 0:
                        guess = Y_hat_str[guess_i]
                        correct = '✓' if guess == Y_str[guess_i] else '✗ (%s)' % Y_str[guess_i]

                        print('%d %d%% (%s) %.4f%% / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start),
                                                               value * 100, guess, correct))

            mean_value = mean_value / n_estimates
            self.log_estimate_results(mean_value)
            if do_print >= 1:
                print('Estimated value = %.2f%%, through  %ix%i estimates (estimating time = %s) ' \
                      % (mean_value * 100, est_batch_size, n_estimates, time_since(start)))

        return mean_value

    def test(self, X_str, batch_first=False):
        vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod

        all_param = self.all_param
        decoder = all_param['decoder']
        encoder = all_param['encoder']

        evaluate = self.evaluate
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        guess_i = 0

        X = tensor_line_from_line(X_str, vocab, EOS_encod, len(X_str)).unsqueeze(batch_dim)

        # evaluating
        Y_hat, output = evaluate(X, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)
        Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)

        return Y_hat_str[guess_i]

    def estimate_some(self, n_estim):
        vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod
        distractor_symbols = self.distractor_symbols

        all_param = self.all_param
        decoder = all_param['decoder']
        encoder = all_param['encoder']

        evaluate = self.evaluate
        gen_random_batch = self.gen_random_batch
        estimate_performance = ArithmeticTask.estimate_performance
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        guess_i = 0
        est_batch_size = 1

        with torch.no_grad():
            main_value = 0
            for epoch in range(1, n_estim + 1):
                # generation
                Y, X = gen_random_batch(est_batch_size, distractor_symbols)

                Y_str = lines_from_tensor_line(Y, vocab_decod, EOS_decod)
                X_str = lines_from_tensor_line(X, vocab, EOS_decod)

                # evaluating
                Y_hat, output = evaluate(X, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)

                # estimating
                Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)
                value = estimate_performance(Y_str, Y_hat_str)
                main_value += value
                guess = Y_hat_str[guess_i]
                correct = '✓' if guess == Y_str[guess_i] else '✗ (%s)' % Y_str[guess_i]
                print('"%s" -> "%s" / %s , value=%.2f%%' % (X_str[guess_i], guess, correct, value * 100))

            main_value = main_value / n_estim
            print('Mean value = %.2f%%, through  %i estimates' % (main_value * 100, n_estim))

        return main_value

    def train(self, Y, X, encoder, decoder, EOS_token_encod, SOS_token_decod, EOS_token_decod,
              batch_first=False):  # X is batch

        # initialising

        all_param = self.all_param
        decod_optimizer = all_param['decod_optimizer']
        encod_optimizer = all_param['encod_optimizer']
        criterion = all_param['criterion']
        device = all_param['device']
        max_grad_norm = all_param['max_grad_norm']

        encoding = self.encoding
        decoding_with_teacher = self.decoding_with_teacher

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        hidden_context = encoding(X, encoder, EOS_token_encod, batch_first)
        output = decoding_with_teacher(hidden_context, decoder, Y, SOS_token_decod, EOS_token_decod)

        Y = Y.to(device)
        loss = 0
        n = Y.size(step_dim)
        for i in range(n):
            loss += criterion(output[i], Y[i])
        loss = loss / n
        loss.backward()
        if max_grad_norm > 0:
            nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)

        decod_optimizer.step()
        encod_optimizer.step()
        return output, loss

    def evaluate(self, X, encoder, decoder, EOS_token_encod, SOS_token_decod, EOS_token_decod, batch_first=False):
        all_param = self.all_param
        device = all_param['device']

        encoding = self.encoding
        decoding = self.decoding

        hidden_context = encoding(X, encoder, EOS_token_encod, batch_first)
        Y_hat, output = decoding(hidden_context, decoder, SOS_token_decod, EOS_token_decod)

        return Y_hat, output

    def encoding(self, X, encoder, EOS_token, batch_first=False):
        all_param = self.all_param
        device = all_param['device']
        find_indexes = self.find_indexes

        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        batch_size = X.size(batch_dim)

        encoder.zero_grad()
        hidden = encoder.init_hidden(batch_size)
        hidden = hidden.to(device)
        X = X.to(device)

        hidden_out = torch.zeros(hidden.size())
        # doing encoder steps
        hidden_list = []
        for i in range(X.size(step_dim)):
            hidden = encoder(X[i], hidden)
            hidden_list.append(hidden)
            hidden = hidden.to(device)

        # taking target output
        out_indexes = find_indexes(X, EOS_token)
        for j in range(batch_size):
            hidden_out[j] = hidden_list[out_indexes[j]][j]

        return hidden_out

    def decoding_with_teacher(self, hidden_context, decoder, Y, SOS_token, EOS_token, batch_first=False):
        all_param = self.all_param
        vocab_decod = self.vocab_decod
        device = all_param['device']
        find_indexes = self.find_indexes

        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0
        size1 = Y.size(1)
        size0 = Y.size(0)
        size2 = len(vocab_decod)
        dim = 0

        batch_size = hidden_context.size(dim)

        decoder.zero_grad()
        hidden = hidden_context
        hidden = hidden.to(device)

        output = torch.zeros((size0, size1, size2))
        X_i = torch.zeros(batch_size, dtype=torch.long)
        X_i.fill_(SOS_token)
        # doing encoder steps
        for i in range(Y.size(step_dim)):
            output_i, hidden = decoder(X_i, hidden)
            output[i] = output_i
            X_i = Y[i]
            hidden = hidden.to(device)
            X_i = X_i.to(device)

        # taking target output

        #         out_indexes = find_indexes(Y,EOS_token)
        #         for j in range(batch_size):
        #             Y_hat[j] = Y_out_list[out_indexes[j]][j]

        return output

    def decoding(self, hidden_context, decoder, SOS_token, EOS_token, batch_first=False):
        all_param = self.all_param
        vocab_decod = self.vocab_decod
        device = all_param['device']
        max_Y_lenght = all_param['max_Y_lenght']

        find_indexes = self.find_indexes
        output_to_tensor_line = AbstractTask.output_to_tensor_line

        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        size1 = 1
        size0 = 1
        size2 = len(vocab_decod)

        dim = 0
        batch_size = 1  # !!!!!!!!!!!!!!!!!!!!

        decoder.zero_grad()
        hidden = hidden_context
        hidden = hidden.to(device)
        output = torch.zeros((size0, size1, size2))
        X_i = torch.zeros(batch_size, dtype=torch.long)
        X_i.fill_(SOS_token)

        output_list = []
        Y_hat_list = []
        # doing encoder steps
        i = 0
        while (X_i.item() != EOS_token) and (i < max_Y_lenght):
            output_i, hidden = decoder(X_i, hidden)
            output_list.append(output_i.unsqueeze(0))
            X_i = output_to_tensor_line(output_i.unsqueeze(0)).squeeze(0)
            Y_hat_list.append(X_i.unsqueeze(0))
            hidden = hidden.to(device)
            X_i = X_i.to(device)
            i += 1

        output = torch.cat(output_list, 0)
        Y_hat = torch.cat(Y_hat_list, 0)

        return Y_hat, output

    def gen_random_batch(self, batch_size, distractor_symbols='', batch_first=False):

        vocab = self.vocab
        vocab_decod = self.vocab_decod
        all_param = self.all_param
        max_X_lenght = all_param['max_X_lenght']
        max_Y_lenght = all_param['max_Y_lenght']
        EOS_token_encod = self.EOS_token_encod
        EOS_token_decod = self.EOS_token_decod

        set_to_training_pair_str = ArithmeticTask.set_to_training_pair_str
        gen_random_set = ArithmeticTask.gen_random_set
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        clipping_tensor = AbstractTask.clipping_tensor

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        X_list = []
        Y_list = []
        for i in range(batch_size):
            X_str, Y_str = set_to_training_pair_str(*gen_random_set(N_max=max_X_lenght), distractor_symbols)

            X_list.append(tensor_line_from_line(X_str, vocab, EOS_token_encod, max_X_lenght).unsqueeze(batch_dim))
            Y_list.append(tensor_line_from_line(Y_str, vocab_decod, EOS_token_decod, max_Y_lenght).unsqueeze(batch_dim))

        X_batch = torch.cat(X_list, 1)
        Y_batch = torch.cat(Y_list, 1)

        X_batch = clipping_tensor(X_batch)
        Y_batch = clipping_tensor(Y_batch)
        return Y_batch, X_batch

    def find_indexes(self, tensor, token, batch_first=False):
        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        step_size = tensor.size(step_dim)
        batch_size = tensor.size(batch_dim)
        indexes = [0 for j in range(batch_size)]
        for j in range(batch_size):
            for i in range(step_size):
                if tensor[i][j] == token:
                    indexes[j] = i
                    break

        return indexes

    def find_2indexes(tensor, token1_tensor, token2_tensor, batch_first=False):

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0
        batch_size = tensor.size(batch_dim)
        i_token1_list = [0 for i in range(batch_size)]
        i_token2_list = [0 for i in range(batch_size)]

        for i in range(batch_size):
            ii = 0
            while (tensor[ii][i].nonzero().size(0) != 0):
                if torch.equal(tensor[ii][i], token1_tensor):
                    i_token1_list[i] = ii
                if torch.equal(tensor[ii][i], token2_tensor):
                    i_token2_list[i] = ii
                ii += 1
                if ii >= tensor.size(step_dim):
                    break

        return i_token1_list, i_token2_list

    # not local function
    def estimate_performance(Y_str, Y_hat_str):
        compare_lines = AbstractTask.compare_lines
        if (len(Y_str) != len(Y_hat_str)):
            print('Error!!! len(Y_str) != len(Y_hat_str)!!')
            return None
        m = len(Y_str)
        values = [compare_lines(Y_str[i], Y_hat_str[i]) for i in range(m)]
        mean = 0
        for value in values: mean += value
        mean = mean / m
        return mean

    def gen_random_set(N_min=10, N_max=30):
        N = random.randint(N_min, N_max)
        n_a = random.randint(2,8)
        n_b = random.randint(2,8)

        a_num_length = 10 ** n_a - 1
        b_num_length = 10 ** n_b - 1
        a = random.randint(-a_num_length, a_num_length)
        b = random.randint(-b_num_length, b_num_length)
        operator = '+' if random.randint(0, 1) else '-'
        return a, b, operator, N

    def set_to_X_str(a, b, operator, end_token='='):
        out_str = ''
        if b >= 0:
            out_str = str(a) + operator + str(b)
        elif operator == '-':
            out_str = str(a) + '+' + str(abs(b))
        else:
            out_str = str(a) + str(b)
        out_str += end_token
        return out_str

    def set_to_Y_hat_str(a, b, operator, end_token='.'):
        out_str = ''
        if operator == '+':
            out_str = str(a + b)
        elif operator == '-':
            out_str = str(a - b)
        out_str += end_token
        return out_str

    def set_to_training_pair_str(a, b, operator, N, distractor=''):
        set_to_X_str = ArithmeticTask.set_to_X_str
        set_to_Y_hat_str = ArithmeticTask.set_to_Y_hat_str

        X_str = set_to_X_str(a, b, operator)
        if distractor != '':
            end_token = X_str[-1]
            out_str = ''
            distractor_len = len(distractor)

            dist = ''
            for i in range(N - len(X_str)):
                dist += distractor[random.randint(0, distractor_len - 1)]
            dist = list(dist)
            target = list(X_str[:-1])
            both = len(target) + len(dist)
            while both > 0:
                p = random.randint(0, both - 1)
                if p < len(target):
                    out_str += target.pop(0)
                else:
                    out_str += dist.pop(0)
                both = len(target) + len(dist)
            out_str += end_token
            X_str = out_str

        Y_str = set_to_Y_hat_str(a, b, operator)
        return X_str, Y_str



class XMLTask(AbstractTask):
    name = 'XML'

    def __init__(self):
        super().__init__()
        return

    def get_rnn(self):
        return (self.all_param['encoder'], self.all_param['decoder'])

    def get_optimizer(self):
        return (self.all_param['encod_optimizer'], self.all_param['decod_optimizer'])

    def initialisation(self, all_parameters):
        # parameters
        super().initialisation(all_parameters)
        create_vocab = AbstractTask.create_vocab
        all_param = self.all_param
        # task param
        self.tag_symbols = string.ascii_lowercase
        self.all_symbols = self.tag_symbols + '<>/.'

        self.vocab, self.EOS_token, self.SOS_token = create_vocab('=')
        self.vocab_decod, self.EOS_token_, self.SOS_token_ = create_vocab(self.all_symbols)

        vocab = self.vocab
        vocab_decod = self.vocab_decod
        self.EOS_token_encod = vocab['=']

        self.SOS_token_decod = vocab_decod['<sos>']
        self.EOS_token_decod = vocab_decod['.']
        # RNN param
        vocab_size = len(self.vocab)
        vocab_decod_size = len(self.vocab_decod)

        hidden_size = all_param['n_hidden']
        device = all_param['device']
        rnn_type = all_param['rnn_type']
        encoder_type = all_param['encoder_type']
        decoder_type = all_param['decoder_type']
        n_layers = all_param['n_layers']
        scale = all_param['scale']

        decoder = decoder_type(vocab_decod_size, hidden_size, rnn_type, scale, n_layers)
        encoder = encoder_type(vocab_size, hidden_size, rnn_type, scale, n_layers)
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        all_param['decoder'] = decoder
        all_param['encoder'] = encoder

        # training param
        criterion = all_param['criterion']
        #         print('Criterion ->', criterion)

        optimizer_name = all_param['optimizer_name']
        learning_rate = all_param['learning_rate']
        if optimizer_name == 'SGD':
            decod_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
            encod_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)

        all_param['decod_optimizer'] = decod_optimizer
        all_param['encod_optimizer'] = encod_optimizer

        # estimating

    def do_train(self, n_epochs=-1, do_print=2):
        #         vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod
        tag_symbols = self.tag_symbols

        all_param = self.all_param

        decoder = all_param['decoder']
        encoder = all_param['encoder']
        batch_size = all_param['batch_size']
        max_X_lenght = all_param['max_X_lenght']
        max_Y_lenght = all_param['max_Y_lenght']
        timemark_print_format = all_param['timemark_print_format']
        print_every = all_param['print_every']
        plot_every = all_param['plot_every']
        refresh_frequency = all_param['refresh_frequency']

        train = self.train
        refresh = self.refresh
        gen_random_batch = self.gen_random_batch
        output_to_tensor_line = AbstractTask.output_to_tensor_line
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        time_since = AbstractTask.time_since

        if n_epochs < 0:
            n_epochs = all_param['n_epochs']

        current_loss = 0
        all_losses = []

        guess_i = 0

        Y_srt_sample = ''
        start = time.time()

        for epoch in range(1, n_epochs + 1):
            # prepare traning_batch and category_batch
            Y_batch, X_batch = gen_random_batch(batch_size, tag_symbols)
            # training
            output, loss = train(Y_batch, X_batch, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)

            Y_srt_sample = lines_from_tensor_line(Y_batch, vocab_decod, EOS_decod)[guess_i]
            del Y_batch
            del X_batch

            current_loss += loss.item()
            if do_print >= 2:
                if epoch % print_every == 0:
                    Y_hat = output_to_tensor_line(output)
                    Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)
                    guess = Y_hat_str[guess_i]
                    correct = '✓' if guess == Y_srt_sample else '✗ (%s)' % Y_srt_sample

                    print('%d %d%% (%s) %.4f / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start),
                                                         loss, guess, correct))

            if epoch % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

            if epoch == n_epochs // refresh_frequency:
                refresh()
                if do_print >= 2:
                    print('Refreshed!')

        stop = time.time()
        learn_time = time_since(start)
        learn_datetime = datetime.datetime.now()
        data_str = learn_datetime.strftime(timemark_print_format)

        if len(all_losses) == 0:
            all_losses += [loss]
        self.log_train_results(n_epochs, all_losses, start, stop)
        if do_print >= 1:
            print("Learning ended. Learn datetime: " + data_str, 'Learn time: ' + str(learn_time))
        if do_print >= 2:
            plt.figure()
            plt.plot(all_losses)
        return

    def do_estimate(self, n_estimates=-1, do_print=2):
        #         vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod
        tag_symbols = self.tag_symbols

        all_param = self.all_param
        decoder = all_param['decoder']
        encoder = all_param['encoder']
        #         batch_size = all_param['batch_size']
        max_X_lenght = all_param['max_X_lenght']
        max_Y_lenght = all_param['max_Y_lenght']
        timemark_print_format = all_param['timemark_print_format']
        print_every = all_param['est_print_every']

        evaluate = self.evaluate
        refresh = self.refresh
        gen_random_batch = self.gen_random_batch
        estimate_performance = ArithmeticTask.estimate_performance
        output_to_tensor_line = AbstractTask.output_to_tensor_line
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        time_since = AbstractTask.time_since

        if n_estimates < 0:
            n_estimates = all_param['n_estimates']

        n_epochs = n_estimates

        est_batch_size = 1
        guess_i = 0

        start = time.time()

        with torch.no_grad():
            mean_value = 0
            for epoch in range(1, n_estimates + 1):
                # generation
                Y, X = gen_random_batch(est_batch_size, tag_symbols)
                Y_str = lines_from_tensor_line(Y, vocab_decod, EOS_decod)

                # evaluating
                Y_hat, output = evaluate(X, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)

                # estimating
                Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)
                value = estimate_performance(Y_str, Y_hat_str)
                mean_value += value
                if do_print >= 2:
                    if epoch % print_every == 0:
                        guess = Y_hat_str[guess_i]
                        correct = '✓' if guess == Y_str[guess_i] else '✗\n (%s)' % Y_str[guess_i]

                        print('%d %d%% (%s) %.4f%% / \nGuess:%s %s' % (epoch, epoch / n_epochs * 100, time_since(start),
                                                                       value * 100, guess, correct))

            mean_value = mean_value / n_estimates
            self.log_estimate_results(mean_value)
            if do_print >= 1:
                print('Estimated value = %.2f%%, through  %ix%i estimates (estimating time = %s) ' \
                      % (mean_value * 100, est_batch_size, n_estimates, time_since(start)))

        return mean_value

    def test(self, X_str, batch_first=False):
        vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod

        all_param = self.all_param
        decoder = all_param['decoder']
        encoder = all_param['encoder']

        evaluate = self.evaluate
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        guess_i = 0

        X = tensor_line_from_line(X_str, vocab, EOS_encod, len(X_str)).unsqueeze(batch_dim)

        # evaluating
        Y_hat, output = evaluate(X, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)
        Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)

        return Y_hat_str[guess_i]

    def estimate_some(self, n_estim):
        vocab = self.vocab
        vocab_decod = self.vocab_decod
        EOS_encod = self.EOS_token_encod
        SOS_decod = self.SOS_token_decod
        EOS_decod = self.EOS_token_decod
        tag_symbols = self.tag_symbols

        all_param = self.all_param
        decoder = all_param['decoder']
        encoder = all_param['encoder']

        evaluate = self.evaluate
        gen_random_batch = self.gen_random_batch
        estimate_performance = ArithmeticTask.estimate_performance
        lines_from_tensor_line = AbstractTask.lines_from_tensor_line

        guess_i = 0
        est_batch_size = 1

        with torch.no_grad():
            main_value = 0
            for epoch in range(1, n_estim + 1):
                # generation
                Y, X = gen_random_batch(est_batch_size, tag_symbols)

                Y_str = lines_from_tensor_line(Y, vocab_decod, EOS_decod)
                X_str = lines_from_tensor_line(X, vocab_decod, EOS_decod)
                # evaluating
                Y_hat, output = evaluate(X, encoder, decoder, EOS_encod, SOS_decod, EOS_decod)

                # estimating
                Y_hat_str = lines_from_tensor_line(Y_hat, vocab_decod, EOS_decod)
                value = estimate_performance(Y_str, Y_hat_str)
                main_value += value
                guess = Y_hat_str[guess_i]
                correct = '✓' if guess == Y_str[guess_i] else '✗\n(%s)' % Y_str[guess_i]
                print('X: "%s" ->\nY_hat: "%s" / %s , value=%.2f%%' % (X_str[guess_i], guess, correct, value * 100))

            main_value = main_value / n_estim
            print('Mean value = %.2f%%, through  %i estimates' % (main_value * 100, n_estim))

        return main_value

    def train(self, Y, X, encoder, decoder, EOS_token_encod, SOS_token_decod, EOS_token_decod,
              batch_first=False):  # X is batch

        # initialising

        all_param = self.all_param
        decod_optimizer = all_param['decod_optimizer']
        encod_optimizer = all_param['encod_optimizer']
        criterion = all_param['criterion']
        device = all_param['device']
        max_grad_norm = all_param['max_grad_norm']

        encoding = self.encoding
        decoding_with_teacher = self.decoding_with_teacher

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        hidden_context = encoding(X, encoder, EOS_token_encod, batch_first)
        output = decoding_with_teacher(hidden_context, decoder, X, SOS_token_decod, EOS_token_decod)

        Y = Y.to(device)
        loss = 0
        n = Y.size(step_dim)
        for i in range(n):
            loss += criterion(output[i], Y[i])

        loss = loss / n

        loss.backward()
        if max_grad_norm > 0:
            #             nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_grad_norm)

        decod_optimizer.step()
        #         encod_optimizer.step()
        return output, loss

    def evaluate(self, X, encoder, decoder, EOS_token_encod, SOS_token_decod, EOS_token_decod, batch_first=False):
        all_param = self.all_param
        device = all_param['device']

        encoding = self.encoding
        decoding = self.decoding

        hidden_context = encoding(X, encoder, EOS_token_encod, batch_first)
        Y_hat, output = decoding(hidden_context, decoder, X, SOS_token_decod, EOS_token_decod)

        return Y_hat, output

    def encoding(self, X, encoder, EOS_token, batch_first=False):
        all_param = self.all_param
        device = all_param['device']

        #         #initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        batch_size = X.size(batch_dim)

        hidden = encoder.init_hidden(batch_size)
        hidden = hidden.to(device)
        hidden_out = hidden

        return hidden_out

    def decoding_with_teacher(self, hidden_context, decoder, X, SOS_token, EOS_token, batch_first=False):
        all_param = self.all_param
        vocab_decod = self.vocab_decod
        device = all_param['device']
        find_indexes = self.find_indexes

        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        size1 = X.size(1)
        size0 = X.size(0)
        size2 = len(vocab_decod)
        dim = 0

        batch_size = hidden_context.size(dim)

        decoder.zero_grad()
        hidden = hidden_context
        hidden = hidden.to(device)

        output = torch.zeros((size0, size1, size2))

        X = X.to(device)
        for i in range(X.size(step_dim)):
            output_i, hidden = decoder(X[i], hidden)
            output[i] = output_i
            hidden = hidden.to(device)
        output = output.to(device)
        return output

    def decoding(self, hidden_context, decoder, X, SOS_token, EOS_token, batch_first=False):
        all_param = self.all_param
        vocab_decod = self.vocab_decod
        device = all_param['device']
        max_Y_lenght = all_param['max_Y_lenght']

        output_to_tensor_line = AbstractTask.output_to_tensor_line

        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        size1 = 1
        size0 = 1
        size2 = len(vocab_decod)

        dim = 0
        batch_size = 1  # !!!!!!!!!!!!!!!!!!!!

        decoder.zero_grad()
        hidden = hidden_context
        hidden = hidden.to(device)
        output = torch.zeros((size0, size1, size2))
        #         X_i = torch.zeros(batch_size, dtype=torch.long)
        #         X_i.fill_(SOS_token)
        X_i = X[0]
        X_i = X_i.to(device)
        max_lenght = X.size(0) + max_Y_lenght
        output_list = []
        Y_hat_list = []
        # doing encoder steps
        i = 0
        while (X_i.item() != EOS_token) and (i < max_lenght):
            output_i, hidden = decoder(X_i, hidden)
            output_list.append(output_i.unsqueeze(0))
            X_i = output_to_tensor_line(output_i.unsqueeze(0)).squeeze(0)
            Y_hat_list.append(X_i.unsqueeze(0))
            hidden = hidden.to(device)
            X_i = X_i.to(device)
            i += 1

        output = torch.cat(output_list, 0)
        Y_hat = torch.cat(Y_hat_list, 0)

        return Y_hat, output

    def gen_random_batch(self, batch_size, tag_symbols='', batch_first=False):

        vocab_decod = self.vocab_decod
        all_param = self.all_param
        EOS_token_decod = self.EOS_token_decod

        line_to_training_pair = self.line_to_training_pair
        get_xml_line = self.get_xml_line
        tensor_line_from_line = AbstractTask.tensor_line_from_line
        clipping_tensor = AbstractTask.clipping_tensor

        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        X_list = []
        Y_list = []
        for i in range(batch_size):
            X_str, Y_str = line_to_training_pair(get_xml_line(tag_symbols))
            X_list.append(tensor_line_from_line(X_str, vocab_decod, EOS_token_decod, len(X_str)).unsqueeze(batch_dim))
            Y_list.append(tensor_line_from_line(Y_str, vocab_decod, EOS_token_decod, len(Y_str)).unsqueeze(batch_dim))

        X_batch = torch.cat(X_list, 1)
        Y_batch = torch.cat(Y_list, 1)

        #         X_batch = clipping_tensor(X_batch)
        #         Y_batch = clipping_tensor(Y_batch)
        return Y_batch, X_batch

    def find_indexes(self, tensor, token, batch_first=False):
        # initialising
        if batch_first:
            batch_dim = 0
            step_dim = 1
        else:
            batch_dim = 1
            step_dim = 0

        step_size = tensor.size(step_dim)
        batch_size = tensor.size(batch_dim)
        indexes = [0 for j in range(batch_size)]
        for j in range(batch_size):
            for i in range(step_size):
                if tensor[i][j] == token:
                    indexes[j] = i
                    break

        return indexes

    # not local function
    def estimate_performance(Y_str, Y_hat_str):
        compare_lines = AbstractTask.compare_lines
        if (len(Y_str) != len(Y_hat_str)):
            print('Error!!! len(Y_str) != len(Y_hat_str)!!')
            return None
        m = len(Y_str)
        values = [compare_lines(Y_str[i], Y_hat_str[i]) for i in range(m)]
        mean = 0
        for value in values: mean += value
        mean = mean / m
        return mean

    def get_xml_line(self, symbol_set, tag_l_min=2, tag_l_max=10, max_iter=5):
        gen_close_tag = self.gen_close_tag
        gen_new_tag = self.gen_new_tag

        tag_list = []
        sep = ''
        tag = gen_new_tag(symbol_set, tag_l_min, tag_l_max)
        tag_list.append(tag)
        xml_line = tag + sep
        i = 0
        while True:
            i += 1
            f = random.randint(0, 1)
            if f or i > max_iter:
                if len(tag_list) == 0:
                    break
                else:
                    last_tag = tag_list.pop(-1)
                    xml_line += gen_close_tag(last_tag) + sep
            else:
                tag = gen_new_tag(symbol_set, tag_l_min, tag_l_max)
                tag_list.append(tag)
                xml_line += tag + sep

            if xml_line[-1] == sep:
                xml_line = xml_line[:-1]

        return xml_line

    def gen_close_tag(self, tag):
        tag_close = '/'
        tag_clse_i = 1
        close_tag = tag[:tag_clse_i] + tag_close + tag[tag_clse_i:]
        return close_tag

    def gen_new_tag(self, symbol_set='qwerty', tag_l_min=2, tag_l_max=10):
        tag_s = '<'
        tag_e = '>'
        set_l = len(symbol_set)
        l = random.randint(tag_l_min, tag_l_max)
        tag = tag_s
        for i in range(l):
            s = symbol_set[random.randint(0, set_l - 1)]
            tag += s
        tag += tag_e
        return tag

    def line_to_training_pair(self, xml_line, eos_symbol='.'):
        X_str = xml_line
        Y_str = X_str[1:] + eos_symbol
        return X_str, Y_str



class FreeTask():
    name = 'freeTask'

    def __init__(self):
        return

    def initialisation(self, all_parameters):
        self.all_param = copy.deepcopy(all_parameters)

    def do_train(self, n_epoch=-1, do_print=2):

        if do_print >= 2:
            print(n_epoch)
        if do_print >= 1:
            print("Train ended!")
        return

    def do_estimate(self, n_epoch=-1, do_print=2):
        value = random.randint(0, 100)  #
        if do_print >= 2:
            print(n_epoch)
        if do_print >= 1:
            print("Estimating ended! Valu = ", value)

        value = value / 100  # не проценты, а доля

        return value

    # дополнительно - наверно просто забей
    def gen_save_dict(self, params):
        save_dict = {'key': 1}
        return save_dict

    def update_all(self, save_dict):

        return


class PTBTask():
    name = 'PTB'

    def __init__(self):
        return

    def initialisation(self, all_parameters):
        self.all_param = copy.deepcopy(all_parameters)

        # all_param
        # model_type = 'LSTM'
        emsize = 400
        nhid = 1150
        # nlayers = 3
        dropout = 0.4
        dropouth = 0.3
        dropouti = 0.65
        dropoute = 0.1
        wdrop = 0.5
        tied = True
        self.clip = 0.25
        eval_batch_size = 10
        test_batch_size = 1
        bptt = 70
        # batch_size = 80
        wdecay = 1.2e-6

        self.bptt = 70
        self.alpha = 2
        self.beta = 1
        self.log_interval = 1

        # load penn treebank dataset
        train, val, test = penn_treebank_dataset(train=True, dev=True, test=True)

        # prepare dataset
        encoder = LabelEncoder(train + val + test)
        ntokens = encoder.vocab_size

        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]

        self.train_data = encoder.batch_encode(train)
        self.val_data = encoder.batch_encode(val)
        self.test_data = encoder.batch_encode(test)

        ntokens = encoder.vocab_size

        self.train_source_sampler, self.val_source_sampler, self.test_source_sampler = tuple(
            [BPTTBatchSampler(d, bptt, self.all_param['batch_size'], True, 'source') for d in (train, val, test)])

        self.train_target_sampler, self.val_target_sampler, self.test_target_sampler = tuple(
            [BPTTBatchSampler(d, bptt, self.all_param['batch_size'], True, 'target') for d in (train, val, test)])

        # model
        model_type = 'GRU'
        rnn_type = self.all_param['rnn_type']
        all_param = self.all_param
        rnn_cell = rnn_type(all_param['n_hidden'], all_param['n_hidden'], all_param['scale'], True)
        model_type = rnn_cell.name
        self.model = RNNModel(
            model_type, ntokens, emsize, nhid, all_param['n_layers'], dropout,
            dropouth, dropouti, dropoute, wdrop, tied
        )
        self.model.to(all_param['device_name'])
        self.params = list(self.model.parameters()) + list(self.all_param['criterion'].parameters())
        # self.all_param['model_params'] = params

        # total_params = sum(
        #    x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size()
        # )
        optimizer_name = all_param['optimizer_name']
        learning_rate = all_param['learning_rate']
        if optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=wdecay)
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=wdecay)

        self.criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)

    def do_train(self, n_epoch=-1, do_print=2):
        # rnn = self.all_param['rnn']
        self.model.train()
        all_param = self.all_param
        losses = []
        current_loss = 0
        total_loss = 0
        all_losses = []
        # batch_size = all_param['batch_size']
        # batch_size = 80
        hidden = self.model.init_hidden(all_param['batch_size'])

        epoch = 0
        start_time = time.time()

        # for epoch in range(1, self.all_param['n_epochs']+1):
        # while epoch < self.all_param['n_epochs']+1:
        while epoch < (n_epoch + 1):
            for source_sample, target_sample in zip(self.train_source_sampler, self.train_target_sampler):
                data = torch.stack([self.train_data[i] for i in source_sample]).t_().contiguous()
                targets = torch.stack([self.train_data[i] for i in target_sample]).t_().contiguous().view(-1)

                hidden = repackage_hidden(hidden)
                self.optimizer.zero_grad()

                output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)
                raw_loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)

                loss = raw_loss
                # Activiation Regularization
                if self.alpha:
                    loss = loss + sum(
                        self.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                if self.beta:
                    loss = loss + sum(
                        self.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:]
                    )
                loss.backward()
                losses.append(loss.item())

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.params, self.clip)
                self.optimizer.step()

                total_loss += raw_loss.item()
                if epoch % self.log_interval == 0 and epoch > 0:
                    cur_loss = total_loss / self.log_interval
                    elapsed = time.time() - start_time
                    if do_print >= 2:
                        print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                              'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                            epoch, epoch,
                            len(self.train_source_sampler) // self.bptt,
                            self.optimizer.param_groups[0]['lr'], elapsed * 1000 / self.log_interval, cur_loss,
                            math.exp(cur_loss), cur_loss / math.log(2)))
                    total_loss = 0
                    start_time = time.time()
                ###
                epoch += 1

                break

        stop = time.time()

        if do_print >= 2:
            print(n_epoch)
        if do_print >= 1:
            print("Train ended!")
        return losses

    def do_estimate(self, n_epoch=-1, do_print=2):
        self.model.eval()
        # if model_ == 'QRNN':
        #    self.model.reset()
        total_loss = 0

        for source_sample, target_sample in zip(self.val_source_sampler, self.val_target_sampler):
            # self.model.train()
            data = torch.stack([self.val_data[i] for i in source_sample])
            targets = torch.stack([self.val_data[i] for i in target_sample]).view(-1)
            with torch.no_grad():
                # print('shapes')
                # print(data.shape)
                batch_size = data.shape[1]
                hidden = self.model.init_hidden(batch_size)
                output, hidden = self.model(data, hidden)
            total_loss += len(data) * self.criterion(self.model.decoder.weight, self.model.decoder.bias, output,
                                                     targets).item()
            hidden = repackage_hidden(hidden)

        value = total_loss / len(self.val_data)

        if do_print >= 2:
            print(n_epoch)
        if do_print >= 1:
            print("Estimating ended! Valu = ", value)

        return value

        '''value = random.randint(0,100) #
        if do_print >= 2:
            print(n_epoch)
        if do_print >= 1:
            print("Estimating ended! Valu = ", value)

        value = value/100 # не проценты, а доля

        return value'''

    # дополнительно - наверно просто забей
    def gen_save_dict(self, params):
        save_dict = {'key': 1}
        return save_dict

    def update_all(self, save_dict):

        return













