from py_nn_999.net import seq2seq_cnn2lstm_net
from py_nn_999.utils import pre_data_utils
import numpy as np
import heapq


def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    _, enc_state_h, enc_state_c = encoder_inference.predict(source)
    state = [enc_state_h, enc_state_c]
    #第一个10,为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,10] = 1

    output = []
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        if char_index == 11:#预测到了终止符则停下来
            break
        output.append(char_index)
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
    return output

# 预测均衡输出
def infer_beam_search(source,encoder_inference, decoder_inference, n_steps, k, display_k):
    # 先通过推理encoder获得预测输入序列的隐状态
    enc_outputs, enc_state_h, enc_state_c = encoder_inference.predict(source)
    state = [enc_state_h, enc_state_c]
    # 第一个10,为起始标志
    predict_seq = [[10]]
    predict_data_list_all = [[[0, predict_seq, state]]]
    output = []
    # 每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps - 1):  # 百位、十位、个位
        predict_data_list_next = []
        predict_data_list = predict_data_list_all[i]
        for j in range(len(predict_data_list)):
            # 给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
            yhat, h, c = decoder_inference.predict([enc_outputs, np.array(predict_data_list[j][1])] + predict_data_list[j][2])
            # 注意，这里的yhat为Dense之后输出的结果，因此与h不同
            state = [h, c]  # 本次状态做为下一次的初始状态继续传递
            list_prob = yhat[0][0].tolist()
            probs_k = heapq.nlargest(k, list_prob)
            for prob_k in probs_k:
                predict_data = []
                predict_data.append(prob_k)  # 第一位保存概率
                predict_data.append([[list_prob.index(prob_k)]])  # 第二位保存索引
                predict_data.append(state)  # 第三位保存状态
                predict_data_list_next.append(predict_data)
        predict_data_list_all.append(predict_data_list_next)
        # display_k = 1  # 需要显示束宽固定为2
        predict_list = heapq.nlargest(display_k, predict_data_list_next, lambda data_list: data_list[0])
        out = []
        for predict in predict_list:
            out.append(predict[1][0][0])
        output.append(out)
    return output

model_train = seq2seq_cnn2lstm_net.Seq2Seq_CNN2LSTM(pre_data_utils.piece_len)
model_train.load_weights("model/seq2seq_cnn2lstm组选.h5")
encoder_infer = seq2seq_cnn2lstm_net.encoder_infer(model_train)
decoder_infer = seq2seq_cnn2lstm_net.decoder_infer(model_train,encoder_infer)

# 实际预测
def predict(k, display_k):
    # 原输入
    source_list,add1_list = pre_data_utils.read_txt_data_predict()
    # 以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
    encoder_input = pre_data_utils.guiyiAndChannel(add1_list)
    out = infer_beam_search(encoder_input, encoder_infer, decoder_infer, pre_data_utils.tgt_out_len, k, display_k)
    print("source_list", source_list)
    print("out", out)

# 测试和验证
def testAndValidate(predict_index=0, k=4, display_k=2):
    # 正确的值
    tgt_in, tgt_out = pre_data_utils.preLabels1(pre_data_utils.read_txt_data_train(), pre_data_utils.piece_len)
    tgt_out = tgt_out[pre_data_utils.test_last_index:][predict_index]
    true_out = tgt_out[0:-1]
    # 原输入
    source_list = pre_data_utils.preDataAdd0(pre_data_utils.read_txt_data_train(), pre_data_utils.piece_len)
    source_list = source_list[pre_data_utils.test_last_index:][predict_index]  # 从倒数第10个样本作为测试样本
    # 原输入+1
    add1_list = pre_data_utils.preDataAdd1(pre_data_utils.read_txt_data_train(), pre_data_utils.piece_len)
    add1_list = add1_list[pre_data_utils.test_last_index:][predict_index]  # 从倒数第10个样本作为测试样本
    # 以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为通道
    encoder_input = pre_data_utils.guiyiAndChannel([add1_list])
    out = infer_beam_search(encoder_input, encoder_infer, decoder_infer, pre_data_utils.tgt_out_len, k, display_k)
    print("source_list", source_list)
    print("true_out", true_out)
    # print("add1_list", add1_list)
    print("out", out)

if __name__ == '__main__':
    # 测试验证
    testAndValidate(26, k=1, display_k=1)
    # 实际预测
    # predict(k=1, display_k=1)




