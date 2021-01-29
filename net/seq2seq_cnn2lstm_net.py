from tensorflow.keras import models
import tensorflow.keras.layers as kl
import tensorflow.keras.backend as k
from tensorflow import keras
from py_nn_999.utils import pre_data_utils


# ----CBAM 卷积注意力网络结构
# 网上原理讲解文章：https://zhuanlan.zhihu.com/p/101590167

# CAM(Channel Attention Module) 通道注意力模型:压缩空间维度
def channel_attention(input_xs, reduction_ratio):  # input_xs (None,piece_len,3,64)
    # 判断输入数据格式，是channels_first还是channels_last
    channel_axis = 1 if k.image_data_format() == "channels_first" else 3
    # get channel 彩色图片为 3
    channel = int(input_xs.shape[channel_axis])  # 64

    maxpool_channel = kl.GlobalMaxPooling2D()(input_xs)  # (None,64) 全局最大池化
    maxpool_channel = kl.Reshape((1, 1, channel))(maxpool_channel)  # ( None,1,1,64)

    avgpool_channel = kl.GlobalAvgPool2D()(input_xs)  # (None,64) 全局平均池化
    avgpool_channel = kl.Reshape((1, 1, channel))(avgpool_channel)  # (None,1,1,64)
    # 权值共享
    dense_one = kl.Dense(units=int(channel * reduction_ratio), activation='relu',
                         kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    dense_two = kl.Dense(units=int(channel), activation='relu',
                         kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = dense_one(maxpool_channel)  # (None,1,1,32)
    mlp_2_max = dense_two(mlp_1_max)  # (None,1,1,64)
    mlp_2_max = kl.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)  # (None,1,1,64)
    # avg path
    mlp_1_avg = dense_one(avgpool_channel)  # (None,1,1,32)
    mlp_2_avg = dense_two(mlp_1_avg)  # (None,1,1,64)
    mlp_2_avg = kl.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)  # (None,1,1,64)

    channel_attention_feature = kl.Add()([mlp_2_max, mlp_2_avg])  # (None,1,1,64)
    channel_attention_feature = kl.Activation('sigmoid')(channel_attention_feature)  # (None,1,1,64)

    multiply_channel_input = kl.Multiply()([channel_attention_feature, input_xs])  # (None,piece_len,3,64)

    return multiply_channel_input

# SAM(Spatial Attention Module) 空间注意力模型：压缩通道维度
def spatial_attention(channel_refined_feature): # channel_refined_feature (None,piece_len,3,64)

    maxpool_spatial = kl.Lambda(lambda x: k.max(x, axis=3, keepdims=True))(channel_refined_feature)  # (None,piece_len,3,1)

    avgpool_spatial = kl.Lambda(lambda x: k.mean(x, axis=3, keepdims=True))(channel_refined_feature)  # (None,piece_len,3,1)

    max_avg_pool_spatial = kl.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])  # (None,piece_len,3,2)

    spatial_attention_feature = kl.Conv2D(filters=1, kernel_size=(2, 2), padding="same", activation='relu')(max_avg_pool_spatial)  # conv(None,piece_len,3,1)

    multiply_channel_spatial = kl.Multiply()([channel_refined_feature, spatial_attention_feature])  # (None,piece_len,3,64)
    return multiply_channel_spatial

# cbam 注意力模块
def cbam_block(input_xs, reduction_ratio=0.5): # input_(None,piece_len,3,64)
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)  # (None,piece_len,3,64)
    spatial_attention_feature = spatial_attention(channel_refined_feature)  # (None,piece_len,3,64)
    add_refined_input = kl.Add()([spatial_attention_feature, input_xs])  # (None,piece_len,3,64)
    return add_refined_input

# 构建CBAM网络模型:即是CNN编码器
def CBAMModel(inputs,piece_len): # inputs (None, piece_len, 3, 1)
    # 卷积层

    x = kl.Conv2D(64, (3, 2), padding='same')(inputs)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(64, (5, 2), padding='same')(x)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)
    #############################################
    x = kl.Conv2D(128, (5, 2), padding='same')(x)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(128, (5, 2), padding='same')(x)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)
    ##############################################
    x = kl.Conv2D(256, (5, 2), padding='same')(x)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)

    x = kl.Conv2D(256, (5, 2), padding='same')(x)
    x = kl.BatchNormalization(axis=3)(x)
    x = kl.Activation('relu')(x)

    # x = kl.Conv2D(256, (5, 2), padding='same')(x)
    # x = kl.BatchNormalization(axis=3)(x)
    # x = kl.Activation('relu')(x)
    #
    # x = kl.Conv2D(256, (5, 2), padding='same')(x)
    # x = kl.BatchNormalization(axis=3)(x)
    # x = kl.Activation('relu')(x)
    #############################################
    # x = kl.Conv2D(512, (5, 2), padding='same')(x)
    # x = kl.BatchNormalization(axis=3)(x)
    # x = kl.Activation('relu')(x)
    #
    # x = kl.Conv2D(512, (5, 2), padding='same')(x)
    # x = kl.BatchNormalization(axis=3)(x)
    # x = kl.Activation('relu')(x)
    #
    # x = kl.Conv2D(512, (5, 2), padding='same')(x)
    # x = kl.BatchNormalization(axis=3)(x)
    # x = kl.Activation('relu')(x)
    #
    # x = kl.Conv2D(512, (5, 2), padding='same')(x)
    # x = kl.BatchNormalization(axis=3)(x)
    # x = kl.Activation('relu')(x)

    # 注意力层
    x = cbam_block(x)  # 调用cbam 注意力模块 (None,piece_len,3,128)

    x = kl.Conv2D(filters=1, kernel_size=(5, 2), padding="same", activation='relu')(x)  # conv(None,piece_len,3,1)

    # 改变形状传给lstm层
    # Reshape和Flatten层的区别：
    # 1、Flatten将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡，Flatten不影响batch的大小
    # 2、Reshape将输出调整为特定形状。
    reshape1 = kl.Reshape((3, piece_len))(x)  # 把行数和列数数互换 !!!!这个地方很关键，应该着重调整 !!!
    return reshape1

# 编码器，标准的RNN/LSTM模型，取最后时刻的隐藏层作为输出
class Encoder(keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_lstm = kl.LSTM(pre_data_utils.hidden_units, activation='tanh', return_sequences=True, return_state=True)# Encode LSTM Layer
    def call(self, inputs):
        outputs, state_h, state_c = self.encoder_lstm(inputs)
        return outputs,state_h, state_c
# 解码器，有三部分输入，一是encoder部分的每个时刻输出，二是encoder的隐藏状态输出，三是decoder的目标输入
class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = kl.Embedding(pre_data_utils.tgt_vocab_size, pre_data_utils.embedding_dim, mask_zero=False)  # Embedding Layer
        self.decoder_lstm = kl.LSTM(pre_data_utils.hidden_units, activation='tanh', return_sequences=True, return_state=True) # Decode LSTM Layer
        self.attention = kl.Attention()  # Attention Layer
        self.concatenate = kl.Concatenate(axis=-1, name='concat_layer')  # Concatenate Layer
    def call(self,enc_outputs,dec_inputs, init_state):
        decoder_embed = self.embedding(dec_inputs)
        dec_outputs, state_h, state_c = self.decoder_lstm(decoder_embed, initial_state=init_state)
        attention_output = self.attention([dec_outputs, enc_outputs])
        concatenate_output = self.concatenate([dec_outputs, attention_output])  # 一定要把注意力输出和解码输出粘接起来
        return concatenate_output, state_h, state_c
# encoder和decoder模块合并，组成一个完整的seq2seq模型
def Seq2Seq_CNN2LSTM(piece_len=3):
    # Encoder Input Layer
    encoder_inputs = kl.Input(shape=(piece_len, 3, 4), name="encode_input")

    cbam_output = CBAMModel(encoder_inputs,piece_len)  # cnn注意力机制提取特征
    # Encoder Layer
    encoder = Encoder()
    enc_outputs, enc_state_h, enc_state_c = encoder(cbam_output)
    enc_states = [enc_state_h, enc_state_c]

    # Decoder Layer
    decoder_inputs = kl.Input(shape=(None,), name="decode_input")
    decoder = Decoder()
    dec_output, _, _ = decoder(enc_outputs,decoder_inputs, enc_states)

    # Dropout Layer
    # dropout_output = kl.Dropout(0.5)(dec_output)

    # Dense Layer
    # dense_outputs = kl.Dense(64, activation='relu', name='fc1')(dec_output)
    # dense_outputs = kl.Dense(64, activation='relu', name='fc2')(dense_outputs)
    dense_outputs = kl.Dense(pre_data_utils.tgt_vocab_size, activation='softmax', name="final_out_dense")(dec_output)
    # seq2seq model
    model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
    return model
# 从seq2seq模型中获取Encoder子模块
def encoder_infer(model):
    input_my = model.get_layer('encode_input').input
    output_my = model.get_layer('encoder').output
    encoder_model = models.Model(inputs=input_my, outputs=output_my)
    return encoder_model
# 从seq2seq模型中获取Decoder子模块，这里没有直接从decoder层取，方便后续decoder的预测推断
def decoder_infer(model,encoder_model):
    encoder_output = encoder_model.get_layer('encoder').output[0]
    maxlen, hidden_units = encoder_output.shape[1:]

    dec_input = model.get_layer('decode_input').input
    enc_output = kl.Input(shape=(maxlen, hidden_units), name='enc_output')
    input_state_h = kl.Input(shape=(hidden_units,))
    input_state_c = kl.Input(shape=(hidden_units,))
    dec_state_input = [input_state_h, input_state_c]  # 上个时刻的状态h,c

    decoder = model.get_layer('decoder')
    dec_outputs, out_state_h, out_state_c = decoder(enc_output, dec_input, dec_state_input)
    dec_states_out = [out_state_h, out_state_c]

    # fc1_dense = model.get_layer('fc1')(dec_outputs)
    # fc2_dense = model.get_layer('fc2')(fc1_dense)

    final_out_dense = model.get_layer('final_out_dense')
    dense_output = final_out_dense(dec_outputs)

    decoder_model = models.Model(inputs=[enc_output,dec_input]+dec_state_input,
                          outputs=[dense_output]+dec_states_out)
    return decoder_model
if __name__ == '__main__':
    model = Seq2Seq_CNN2LSTM(0)
    model.summary()