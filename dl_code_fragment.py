#coding:utf-8
#based on tensorflow and keras
#by huanggaopan 2018.2

import keras
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.callbacks import TensorBoard, LambdaCallback, CSVLogger


####�����������ṹ���ɼ�Ϊ�����뵥�����������������
#Ӧ�ó���������ͼ����Ϣ�����������������Ϊ�����룬������������һ�εײ�ṹ����ͬ������Ȩ�ز�ͬ��

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
main_input = Input(shape=(100,), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)

auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})
model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
model.evaluate({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          batch_size=128)
rst = model.predict({'main_input': headline_data, 'aux_input': additional_data})


####��������ṹ
tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))

shared_lstm = LSTM(64)

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)
predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=10)


####ͨ���ڵ�����output��ȡ�������м�ڵ��ֵ���Ա�����㷨
layer.output
#�����shape
layer.output_shape
#���ڹ���ṹ��һ��������Ӧ�������ʱ
layer.get_output_at(0)
layer.get_output_at(1)
layer.get_input_shape_at(0)
layer.get_output_shape_at(0)


####���ܽ�����ӻ�
class BatchPerf(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
batPerf = BatchPerf()

def prt(batch, logs):
    print '\nbatch %d train loss: %.4f' % (batch, logs.get('loss'))
batch_print_callback = LambdaCallback(on_batch_end=prt)#ÿ��batch������ִ�д�ӡ

tensorboard = TensorBoard(log_dir='./results/resnetlogs')
csv_logger = CSVLogger('./results/epoch_perf.log')##��¼epoch,acc,loss,val_acc,val_loss

print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
###his.history��¼����Ϣ��CSVLogger��¼��Ϣ��һ��
his = model.fit_generator(generator=generate_train(32),steps_per_epoch=train_num/32,
    epochs=2,callbacks=[tensorboard,csv_logger,batPerf,batch_print_callback],
    validation_data=generate_val(32), validation_steps=val_num/32)
print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

loss_df = pd.DataFrame([batPerf.losses, batPerf.acc]).T
loss_df.to_csv('./results/batch_perf.log',header=['loss', 'acc'],index=None)

####ģ�ͽṹ���ӻ�
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='./model_arch.png')

####LSTM��ֵԤ��ṹ
#lstm����return_sequences=True��ͨ��layer.output���Ի�ȡtimestep�������
#������Ҫ��ȡ��Ӧλ�õ������


#####model save and load
####save model
#serialize model to JSON, the representation of JSON string does not 
#include the weights, only the architecture. 
print("Saved model architecture to disk")
model_json = model.to_json()
with open("model_arch.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5, saves the weights of the model as a HDF5 file.
print("Saved model weights to disk")
model.save_weights("model_w.h5")

####load model
from keras.models import model_from_json
# load json and create model
print("Loaded model architecture from disk")
json_file = open('model_arch.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
print("Loaded model weights from disk")
loaded_model.load_weights("model.h5")



