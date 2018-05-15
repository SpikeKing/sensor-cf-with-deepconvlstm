# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/4/18
"""
import os
from keras import Input, Model
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Dropout, LSTM, Concatenate, Activation
from keras.optimizers import Adam
from keras.utils import plot_model

from bases.model_base import ModelBase


class DclModel(ModelBase):
    """
    DeepConvLSTM模型
    """

    def __init__(self, config):
        super(DclModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = self.dcl_model()
        plot_model(self.model, to_file=os.path.join(self.config.img_dir, "model.png"), show_shapes=True)  # 绘制模型图

    @staticmethod
    def dcl_model():
        n_classes = 6
        kernel_size = 3
        f_act = 'relu'
        pool_size = 2
        dropout_rate = 0.15

        # 三个子模型的输入数据
        main_input1 = Input(shape=(128, 3), name='main_input1')
        main_input2 = Input(shape=(128, 3), name='main_input2')
        main_input3 = Input(shape=(128, 3), name='main_input3')

        def cnn_lstm_cell(main_input):
            """
            基于DeepConvLSTM算法, 创建子模型
            :param main_input: 输入数据
            :return: 子模型
            """
            sub_model = Conv1D(512, kernel_size, input_shape=(128, 3), activation=f_act, padding='same')(main_input)
            sub_model = BatchNormalization()(sub_model)
            sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
            sub_model = Dropout(dropout_rate)(sub_model)
            sub_model = Conv1D(64, kernel_size, activation=f_act, padding='same')(sub_model)
            sub_model = BatchNormalization()(sub_model)
            sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
            sub_model = Dropout(dropout_rate)(sub_model)
            sub_model = Conv1D(32, kernel_size, activation=f_act, padding='same')(sub_model)
            sub_model = BatchNormalization()(sub_model)
            sub_model = MaxPooling1D(pool_size=pool_size)(sub_model)
            sub_model = LSTM(128, return_sequences=True)(sub_model)
            sub_model = LSTM(128, return_sequences=True)(sub_model)
            sub_model = LSTM(128)(sub_model)
            main_output = Dropout(dropout_rate)(sub_model)
            return main_output

        first_model = cnn_lstm_cell(main_input1)
        second_model = cnn_lstm_cell(main_input2)
        third_model = cnn_lstm_cell(main_input3)

        model = Concatenate()([first_model, second_model, third_model])  # 合并模型
        model = Dropout(0.4)(model)
        model = Dense(n_classes)(model)
        model = BatchNormalization()(model)
        output = Activation('softmax', name="softmax")(model)

        model = Model([main_input1, main_input2, main_input3], output)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        return model
