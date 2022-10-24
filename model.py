import setup, tensorflow as tf
from data_generator import MyGenerator
from efficientdet.augmentor.misc import MiscEffect
from efficientdet.model import efficientdet
from efficientdet.losses import smooth_l1, focal

phi = 4
batch_size = 4
train_generator = MyGenerator(data_dir='./caer/train', set_name='sampling annotations train.json', batch_size=batch_size, phi=phi, misc_effect=MiscEffect())
validation_generator = MyGenerator(data_dir='./caer/test', set_name='sampling annotations test.json', batch_size=batch_size, phi=phi)
model, _ = efficientdet(phi, num_classes=train_generator.num_classes(), weighted_bifpn=True)
model.load_weights('./checkpoint/model weights.h5', by_name=True, skip_mismatch=True)
# freeze backbone layers
# for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
# 	model.layers[i].trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(), loss={'regression': smooth_l1(), 'classification': focal()}, run_eagerly=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint('./checkpoint/model weights.h5', monitor='loss', save_best_only=True, save_weights_only=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')
model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[checkpoint, reduce_lr])