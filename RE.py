import keras
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras import Sequential
from keras.layers import Input
from keras.layers import Dense

def main():
    with open('data2/train_data.txt', encoding="utf-8") as f:
        a = f.readlines()
        # print(a[:4])
    training_data = []
    for i in a:
        i.replace("\n", "")
        training_data.append(i.split())
    print(training_data[:3])






    with open('data2/train_label.txt', encoding="utf-8") as f:
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



    tag_to_ix = {'未知': 0, '竞争': 1, '隶属': 2, '上下级': 3, '同级': 4, '夫妻': 5, '亲属': 6, 'pad': 7}



    vec_data = []
    vocab_data = []
    for sentence in training_data[:]:
        for c in sentence:
            if c in word_to_ix:
                vocab_data.append(word_to_ix[c])
        vec_data.append(vocab_data)
        vocab_data = []




    max_length = max(len(i) for i in training_data)
    matrix = keras.preprocessing.sequence.pad_sequences(vec_data, maxlen=max_length, padding='post', value=0)
    print(matrix[:3])
    m = np.array(matrix)
    print(m.shape)





    #建立模型
    model = Sequential()

    # embedding 层
    model.add(Embedding(input_dim=len(word_to_ix), output_dim=100, mask_zero=True))  # Random embedding



    model.add(LSTM(output_dim=100, activation='tanh'))
    model.add(Dense(7, activation='softmax'))

    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    matrix_2 = np.reshape(training_label, (50, 1))

    model.fit(matrix, matrix_2, batch_size=32, epochs=50)




    #测试
    with open('data2/dev_1.txt', encoding="utf-8") as f:
        c = f.readlines()

    test_data = []
    for i in c:
        i.replace("\n", "")
        test_data.append(i.split())
    print(test_data[:])


    test_list = []
    test_vec = []
    for sentence in test_data[:]:
        for c in sentence:
            if c in word_to_ix:
                test_list.append(word_to_ix[c])
        test_vec.append(test_list)
        test_list = []

    ans = model.predict_classes(test_vec)
    print(ans)

if __name__ == '__main__':
    main()