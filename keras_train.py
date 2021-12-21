from keras.optimizers import Adam, SGD, Adamax, Adadelta
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
import os
from network.keras_model import xu_JNet, SRNet
from dataset.keras_dataset import my_dataset_generator
from keras.utils import multi_gpu_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

model_name = 'xu_JNet'
if model_name == 'xu_JNet':
    model = xu_JNet(input_shape=(512, 512, 1))
elif model_name == 'SRNet':
    model = SRNet(input_shape=(512, 512, 1))
# model = multi_gpu_model(model, gpus=6)
model_path = None

try:
    model.load_weights(model_path)  # 加载模型
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train = True
if train:
    # adadelta = Adadelta(lr=0.001, decay=0.0005)
    sgd = SGD(lr=0.0001, momentum=0.99, decay=0.0001, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='./logs/' + model_name, write_graph=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factoc=0.99, mode='auto')
    checkpoint = ModelCheckpoint(filepath='./output/keras_weights/' + model_name + '_dropout--{epoch:02d}--{val_loss:.4f}.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 period=1)

    batch_size = 16
    num_val = 45000
    num_train = 405000
    cover_dir = '/raid/chenby/alaska2/Cover'
    stego_dir = '/raid/chenby/alaska2/JMiPOD'
    history = model.fit_generator(generator=my_dataset_generator(batch_size=batch_size,
                                                                 cover_dir=cover_dir,
                                                                 stego_dir=stego_dir,
                                                                 channel_index=0),
                                  steps_per_epoch=max(1, num_train // batch_size),
                                  validation_data=my_dataset_generator(batch_size=batch_size,
                                                                       cover_dir=cover_dir,
                                                                       stego_dir=stego_dir,
                                                                       train_set=False,
                                                                       channel_index=0),
                                  validation_steps=max(1, num_val // batch_size),
                                  epochs=20,
                                  callbacks=[tensorboard, checkpoint, reduce_lr], #
                                  initial_epoch=0,
                                  shuffle=False)

    # Visualization(history=history, model_name='xu_JNet_dropout')




