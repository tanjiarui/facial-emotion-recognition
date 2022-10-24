import config, setup, tensorflow as tf, tf2onnx
from efficientdet.model import efficientdet

model_weights = config.arguments.get('model weights')
phi = 4
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
image_size = image_sizes[phi]
score_threshold = config.arguments.get('confidence threshold')
_, model = efficientdet(phi, num_classes=7, weighted_bifpn=True, score_threshold=score_threshold)
model.load_weights(model_weights, by_name=True, skip_mismatch=True)
spec = [tf.TensorSpec((None, image_size, image_size, 3), tf.float16, name='input')]
output_path = model.name + '.onnx'
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
for n in model_proto.graph.output:
	print(n.name)