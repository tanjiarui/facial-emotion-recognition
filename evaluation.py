import config, setup, os
from data_generator import MyGenerator
from efficientdet.model import efficientdet
from efficientdet.eval.coco import evaluate

data_root = config.arguments.get('dataset')
test_root = os.path.join(data_root, 'test')
model_weights = config.arguments.get('model weights')
phi = 4
batch_size = config.arguments.get('batch size')
score_threshold = config.arguments.get('confidence threshold')
validation_generator = MyGenerator(data_dir=test_root, set_name='annotations test.json', batch_size=batch_size, phi=phi)
_, model = efficientdet(phi, num_classes=validation_generator.num_classes(), weighted_bifpn=True, score_threshold=score_threshold)
model.load_weights(model_weights, by_name=True, skip_mismatch=True)
evaluate(validation_generator, model, threshold=score_threshold)