import setup, config, numpy as np, cv2, tensorflow as tf, matplotlib.cm as cm
from efficientdet.utils import preprocess_image
from efficientdet.model import efficientdet

phi = 4
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
image_size = image_sizes[phi]


def load_image(path):
	image = cv2.imread(path)
	if image is None:
		print('no such file ', path)
		exit()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image, _ = preprocess_image(image, image_size)
	return image


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
	# First, we create a model that maps the input image to the activations
	# of the last conv layer as well as the output predictions
	grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
	# Then, we compute the gradient of the top predicted class for our input image
	# with respect to the activations of the last conv layer
	with tf.GradientTape() as tape:
		last_conv_layer_output, detections = grad_model(img_array)
		_, scores, _ = detections
		# if not pred_index:
		# 	pred_index = tf.argmax(scores)

	# This is the gradient of the output neuron (top predicted or chosen)
	# with regard to the output feature map of the last conv layer
	grads = tape.gradient(scores, last_conv_layer_output)

	# This is a vector where each entry is the mean intensity of the gradient
	# over a specific feature map channel
	pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

	# We multiply each channel in the feature map array
	# by "how important this channel is" with regard to the top predicted class
	# then sum all the channels to obtain the heatmap class activation
	last_conv_layer_output = last_conv_layer_output[0]
	heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
	heatmap = tf.squeeze(heatmap)

	# For visualization purpose, we will also normalize the heatmap between 0 & 1
	heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
	return heatmap.numpy()


def save_gradcam(path, heatmap, cam_path='grad cam.jpg', alpha=0.4):
	# load the original image
	image = tf.keras.preprocessing.image.load_img(path)
	image = tf.keras.preprocessing.image.img_to_array(image)
	heatmap = np.uint8(255 * heatmap)  # de-normalization
	jet = cm.get_cmap('jet')  # colorize heatmap
	jet_colors = jet(np.arange(256))[:, :3]
	jet_heatmap = jet_colors[heatmap]
	# create an image with RGB colorized heatmap
	jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
	jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
	jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
	# superimpose the heatmap on original image
	superimposed_image = jet_heatmap * alpha + image
	superimposed_image = tf.keras.preprocessing.image.array_to_img(superimposed_image)
	# save the superimposed image
	superimposed_image.save(cam_path)
	print('the heatmap is saved to ', cam_path)


path = config.arguments.get('image path')
preprocessed_input = load_image(path)
score_threshold = config.arguments.get('confidence threshold')
model_weights = config.arguments.get('model weights')
classes = {1: 'angry', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
_, model = efficientdet(phi, num_classes=len(classes), weighted_bifpn=True, score_threshold=score_threshold)
model.load_weights(model_weights, by_name=True, skip_mismatch=True)
heatmap = make_gradcam_heatmap(img_array=np.expand_dims(preprocessed_input, axis=0), model=model, last_conv_layer_name='fpn_cells/cell_6/fnode7/op_after_combine12/conv')
save_gradcam(path, heatmap)