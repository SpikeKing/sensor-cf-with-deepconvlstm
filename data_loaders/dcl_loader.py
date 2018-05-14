# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os

import numpy as np
from keras.utils import to_categorical

from bases.data_loader_base import DataLoaderBase
from root_dir import ROOT_DIR


class DclLoader(DataLoaderBase):
    def __init__(self, config=None):
        super(DclLoader, self).__init__(config)
        data_path = os.path.join(ROOT_DIR, "data", "UCI_HAR_Dataset")

        self.X_train, self.y_train, self.X_test, self.y_test = self.__load_data(data_path)

        print "[INFO] X_train[0].shape: %s, y_train.shape: %s" \
              % (str(self.X_train[0].shape), str(self.y_train.shape))
        print "[INFO] X_test[0].shape: %s, y_test.shape: %s" \
              % (str(self.X_test[0].shape), str(self.y_test.shape))

    @staticmethod
    def __load_data(data_path):
        """
        加载本地的UCI的训练数据和验证数据
        :param data_path 数据集
        :return: 训练数据和验证数据
        """
        train_path = os.path.join(data_path, "train")
        train_X_path = os.path.join(train_path, "Inertial Signals")

        X_trainS1_x = np.loadtxt(os.path.join(train_X_path, "body_acc_x_train.txt"))
        X_trainS1_y = np.loadtxt(os.path.join(train_X_path, "body_acc_y_train.txt"))
        X_trainS1_z = np.loadtxt(os.path.join(train_X_path, "body_acc_z_train.txt"))
        X_trainS1 = np.array([X_trainS1_x, X_trainS1_y, X_trainS1_z])
        X_trainS1 = X_trainS1.transpose([1, 2, 0])

        X_trainS2_x = np.loadtxt(os.path.join(train_X_path, "body_gyro_x_train.txt"))
        X_trainS2_y = np.loadtxt(os.path.join(train_X_path, "body_gyro_y_train.txt"))
        X_trainS2_z = np.loadtxt(os.path.join(train_X_path, "body_gyro_z_train.txt"))
        X_trainS2 = np.array([X_trainS2_x, X_trainS2_y, X_trainS2_z])
        X_trainS2 = X_trainS2.transpose([1, 2, 0])

        X_trainS3_x = np.loadtxt(os.path.join(train_X_path, "total_acc_x_train.txt"))
        X_trainS3_y = np.loadtxt(os.path.join(train_X_path, "total_acc_y_train.txt"))
        X_trainS3_z = np.loadtxt(os.path.join(train_X_path, "total_acc_z_train.txt"))
        X_trainS3 = np.array([X_trainS3_x, X_trainS3_y, X_trainS3_z])
        X_trainS3 = X_trainS3.transpose([1, 2, 0])

        Y_train = np.loadtxt(os.path.join(train_path, "y_train.txt"))
        Y_train = to_categorical(Y_train - 1.0)  # 标签是从1开始

        print "训练数据: "
        print "传感器1: %s, 传感器1的X轴: %s" % (str(X_trainS1.shape), str(X_trainS1_x.shape))
        print "传感器2: %s, 传感器2的X轴: %s" % (str(X_trainS2.shape), str(X_trainS2_x.shape))
        print "传感器3: %s, 传感器3的X轴: %s" % (str(X_trainS3.shape), str(X_trainS3_x.shape))
        print "传感器标签: %s" % str(Y_train.shape)
        print ""

        test_path = os.path.join(data_path, "test")
        test_X_path = os.path.join(test_path, "Inertial Signals")

        X_valS1_x = np.loadtxt(os.path.join(test_X_path, "body_acc_x_test.txt"))
        X_valS1_y = np.loadtxt(os.path.join(test_X_path, "body_acc_y_test.txt"))
        X_valS1_z = np.loadtxt(os.path.join(test_X_path, "body_acc_z_test.txt"))
        X_valS1 = np.array([X_valS1_x, X_valS1_y, X_valS1_z])
        X_valS1 = X_valS1.transpose([1, 2, 0])

        X_valS2_x = np.loadtxt(os.path.join(test_X_path, "body_gyro_x_test.txt"))
        X_valS2_y = np.loadtxt(os.path.join(test_X_path, "body_gyro_y_test.txt"))
        X_valS2_z = np.loadtxt(os.path.join(test_X_path, "body_gyro_z_test.txt"))
        X_valS2 = np.array([X_valS2_x, X_valS2_y, X_valS2_z])
        X_valS2 = X_valS2.transpose([1, 2, 0])

        X_valS3_x = np.loadtxt(os.path.join(test_X_path, "total_acc_x_test.txt"))
        X_valS3_y = np.loadtxt(os.path.join(test_X_path, "total_acc_y_test.txt"))
        X_valS3_z = np.loadtxt(os.path.join(test_X_path, "total_acc_z_test.txt"))
        X_valS3 = np.array([X_valS3_x, X_valS3_y, X_valS3_z])
        X_valS3 = X_valS3.transpose([1, 2, 0])

        Y_val = np.loadtxt(os.path.join(test_path, "y_test.txt"))
        Y_val = to_categorical(Y_val - 1.0)

        print "验证数据: "
        print "传感器1: %s, 传感器1的X轴: %s" % (str(X_valS1.shape), str(X_valS1.shape))
        print "传感器2: %s, 传感器2的X轴: %s" % (str(X_valS2.shape), str(X_valS2.shape))
        print "传感器3: %s, 传感器3的X轴: %s" % (str(X_valS3.shape), str(X_valS3.shape))
        print "传感器标签: %s" % str(Y_val.shape)
        print "\n"

        return [X_trainS1, X_trainS2, X_trainS3], Y_train, [X_valS1, X_valS2, X_valS3], Y_val

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test


if __name__ == '__main__':
    dl = DclLoader()
