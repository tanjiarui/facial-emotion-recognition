import config, setup, os, time, numpy as np, onnxruntime as ort
from efficientdet.utils import preprocess_image, postprocess_boxes
from efficientdet.utils.draw_boxes import *


def run(raw_video: os.path, annotated_video: os.path):
	capture = cv2.VideoCapture(raw_video)
	width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	writer = cv2.VideoWriter(annotated_video, cv2.VideoWriter_fourcc(*'mp4v'), capture.get(cv2.CAP_PROP_FPS), (width, height))
	fps = 0
	while True:
		ret, image = capture.read()
		if not ret:
			break
		src_image = image.copy()
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# resize and normalization
		image, scale = preprocess_image(image, image_size=image_size)
		# run model
		start = time.time()
		boxes, scores, labels = model.run(None, {'input': np.expand_dims(image, axis=0)})
		boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
		fps = (fps + (1 / (time.time() - start))) / 2
		print('fps = %.2f' % fps)
		# filter by threshold
		indices = np.where(scores[:] > score_threshold)[0]
		boxes = boxes[indices]
		labels = labels[indices]
		# resize bounding boxes
		boxes = postprocess_boxes(boxes=boxes, scale=scale, height=height, width=width)
		draw_boxes(src_image, boxes, scores, labels, colors, classes)
		# cv2.imshow('image', src_image)
		writer.write(src_image)
		if cv2.waitKey(1) & 0xff == ord('q'):
			break

	capture.release()
	writer.release()
	cv2.destroyAllWindows()


phi = 4
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
image_size = image_sizes[phi]
classes = {1: 'angry', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprise'}
colors = [np.random.randint(0, 256, 3).tolist() for _ in range(1, len(classes) + 2)]
score_threshold = config.arguments.get('confidence threshold')
# load the fer model
model = ort.InferenceSession('efficientdet_p.onnx')
run(config.arguments.get('raw video'), config.arguments.get('output'))