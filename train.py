import config, setup, os, tensorflow as tf
from data_generator import MyGenerator
from efficientdet.augmentor.misc import MiscEffect
from efficientdet.model import efficientdet
from efficientdet.losses import smooth_l1, focal

data_root = config.arguments.get('dataset')
train_root = os.path.join(data_root, 'train')
test_root = os.path.join(data_root, 'test')
model_weights = config.arguments.get('model weights')
phi = 4
batch_size = config.arguments.get('batch size')
epochs = config.arguments.get('epochs')
train_generator = MyGenerator(data_dir=train_root, set_name='annotations train.json', batch_size=batch_size, phi=phi, misc_effect=MiscEffect())
validation_generator = MyGenerator(data_dir=test_root, set_name='annotations test.json', batch_size=batch_size, phi=phi)
model, _ = efficientdet(phi, num_classes=train_generator.num_classes(), weighted_bifpn=True)
if model_weights:
	model.load_weights(model_weights, by_name=True, skip_mismatch=True)
	print('weights loaded')
# freeze backbone layers
# for i in range(1, [227, 329, 329, 374, 464, 566, 656][phi]):
# 	model.layers[i].trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(), loss={'regression': smooth_l1(), 'classification': focal(gamma=4)})

checkpoint = tf.keras.callbacks.ModelCheckpoint(model_weights, monitor='val_loss', save_best_only=True, save_weights_only=True)
tensorboard = tf.keras.callbacks.TensorBoard()
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, mode='auto')
model.fit(train_generator, epochs=epochs, validation_data=validation_generator, callbacks=[checkpoint, tensorboard, reduce_lr])