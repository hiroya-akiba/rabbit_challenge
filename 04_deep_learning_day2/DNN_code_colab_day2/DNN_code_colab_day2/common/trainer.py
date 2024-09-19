import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common import optimizer

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, d_train, x_test, d_test, epochs=20, mini_batch_size=100, optimizer_name='SGD', optimizer_param={'learning_rate':0.01}, 
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.d_train = d_train
        self.x_test = x_test
        self.d_test = d_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':optimizer.SGD, 'momentum':optimizer.Momentum, 'nesterov':optimizer.Nesterov,
                                'adagrad':optimizer.AdaGrad, 'rmsprpo':optimizer.RMSprop, 'adam':optimizer.Adam}
        self.optimizer = optimizer_class_dict[optimizer_name.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        d_batch = self.d_train[batch_mask]
        
        grad = self.network.gradient(x_batch, d_batch)
        self.optimizer.update(self.network.params, grad)
        
        loss = self.network.loss(x_batch, d_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print("train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            x_train_sample, d_train_sample = self.x_train, self.d_train
            x_test_sample, d_test_sample = self.x_test, self.d_test
            if not self.evaluate_sample_num_per_epoch is None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, d_train_sample = self.x_train[:t], self.d_train[:t]
                x_test_sample, d_test_sample = self.x_test[:t], self.d_test[:t]
                
            train_acc = self.network.accuracy(x_train_sample, d_train_sample)
            test_acc = self.network.accuracy(x_test_sample, d_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.d_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

