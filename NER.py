import keras
import numpy as np
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras import Sequential
from keras_contrib.layers import CRF
from keras.preprocessing.sequence import pad_sequences

def main():
    #载入训练集数据
    with open('data\dev.txt', encoding="utf-8") as f:
        a = f.readlines()
        # print(a[:4])
    training_data = []
    for i in a:
        i.replace("\n", "")
        training_data.append(i.split())
    print(training_data[:5])


    #载入训练集标签
    with open('data\dev-lable.txt') as f:
        b = f.readlines()
        # print(b[:5])

    training_label = []

    for i in b:
        i.replace("\n", "")
        training_label.append(i.split())
    print(training_label[:5])

    # 建立词典
    word_to_ix = {"pad": 0}
    for sentence in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)


    #建立标签集
    tag_to_ix = {'O': 0, 'B-LOC': 1, 'I-LOC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'pad': 7}




    #构建数据词向量vec_data
    vec_data = []
    vocab_data = []
    for sentence in training_data[:]:
        for c in sentence:
            if c in word_to_ix:
                vocab_data.append(word_to_ix[c])
        vec_data.append(vocab_data)
        vocab_data = []
    print(vec_data[:3])


    #构建标签词向量vocab_label
    vec_label = []
    vocab_label = []
    for sentence in training_label[:]:
        for c in sentence:
            if c in tag_to_ix:
                vocab_label.append(tag_to_ix[c])
        vec_label.append(vocab_label)
        vocab_label = []
    print(vec_label[:3])



    #填充数据词向量
    max_length = max(len(i) for i in training_data)
    matrix = keras.preprocessing.sequence.pad_sequences(vec_data, maxlen=max_length, padding='post', value=0)
    print(matrix[:3])
    m = np.array(matrix)
    print(m.shape)


    #填充标签词向量
    max_length = max(len(i) for i in training_label)
    print(max_length)
    matrix_label = keras.preprocessing.sequence.pad_sequences(vec_label, maxlen=max_length, padding='post', value=7)
    print(matrix_label[:3])
    print(matrix_label.shape)



    #建立模型
    model = Sequential()
    # embedding 层
    model.add(Embedding(input_dim=len(word_to_ix), output_dim=100, mask_zero=True))  # Random embedding
    # bilstm 层
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    # crf 层
    crf = CRF(len(tag_to_ix), sparse_target=True)
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    #训练
    matrix_2 = np.expand_dims(matrix_label, 2)
    model.fit(matrix, matrix_2, batch_size=32, epochs=30)




    # 测试
    with open('data\dev_1.txt', encoding="utf-8") as f:
        c = f.readlines()
    test_data = []
    for i in c:
        i.replace("\n", "")
        test_data.append(i.split())
    print(test_data[:])


    # 构建测试数据词向量vec_data
    test_list = []
    test_vec = []
    for sentence in test_data[:]:
        for c in sentence:
            if c in word_to_ix:
                test_list.append(word_to_ix[c])
        test_vec.append(test_list)
        test_list = []



    #填充测试数据
    input_data = pad_sequences(test_vec, 236, padding='post')

    #测试
    ans = model.predict(test_vec)

    ans = np.squeeze(ans)

    new_tag_to_ix = {v: k for k, v in tag_to_ix.items()}

    new_ans = []
    for i in range(len(ans)):
        tmp = np.argmax(ans[i])
        new_ans.append(new_tag_to_ix[tmp])
    print(new_ans)


if __name__ == '__main__':
    main()





