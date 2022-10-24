import os, cv2, numpy as np
from efficientdet.generators.common import Generator
from pycocotools.coco import COCO


def preprocess_image(image: np.array):
	image = image.astype('float')
	image /= 255
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	image -= mean
	image /= std
	return image


def postprocess_boxes(boxes, height, width):
	c_boxes = boxes.copy()
	c_boxes[:, 0] = np.clip(c_boxes[:, 0], 0, width - 1)
	c_boxes[:, 1] = np.clip(c_boxes[:, 1], 0, height - 1)
	c_boxes[:, 2] = np.clip(c_boxes[:, 2], 0, width - 1)
	c_boxes[:, 3] = np.clip(c_boxes[:, 3], 0, height - 1)
	return c_boxes


class MyGenerator(Generator):
	def __init__(self, data_dir, set_name, **kwargs):
		self.data_dir = data_dir
		self.classes = dict()
		self.coco_labels = dict()
		self.coco_labels_inverse = dict()
		self.labels = dict()
		self.coco = COCO(set_name)
		self.image_ids = self.coco.getImgIds()
		self.load_classes()
		super(MyGenerator, self).__init__(**kwargs)

	def load_classes(self):
		categories = self.coco.loadCats(self.coco.getCatIds())
		categories.sort(key=lambda x: x['id'])
		for c in categories:
			self.coco_labels[len(self.classes)] = c['id']
			self.coco_labels_inverse[c['id']] = len(self.classes)
			self.classes[c['name']] = len(self.classes)
		for key, value in self.classes.items():
			self.labels[value] = key

	def size(self):
		return len(self.image_ids)

	def num_classes(self):
		return len(self.classes)

	def has_label(self, label):
		return label in self.labels

	def has_name(self, name):
		return name in self.classes

	def name_to_label(self, name):
		return self.classes[name]

	def label_to_name(self, label):
		return self.labels[label]

	def coco_label_to_label(self, coco_label):
		return self.coco_labels_inverse[coco_label]

	def coco_label_to_name(self, coco_label):
		return self.label_to_name(self.coco_label_to_label(coco_label))

	def label_to_coco_label(self, label):
		return self.coco_labels[label]

	def image_aspect_ratio(self, image_index):
		image = self.coco.loadImgs(self.image_ids[image_index])[0]
		return float(image['width']) / float(image['height'])

	def load_image(self, image_index):
		image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
		path = os.path.join(self.data_dir, image_info['file_name'])
		image = cv2.imread(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		return image

	def load_annotations(self, image_index):
		annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
		annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32)}
		if len(annotations_ids) == 0:
			return annotations
		coco_annotations = self.coco.loadAnns(annotations_ids)
		for idx, a in enumerate(coco_annotations):
			# some annotations have basically no width / height, skip them
			if a['bbox'][2] < 1 or a['bbox'][3] < 1:
				continue
			annotations['labels'] = np.concatenate(
				[annotations['labels'], [a['category_id'] - 1]], axis=0)
			annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
				a['bbox'][0],
				a['bbox'][1],
				a['bbox'][0] + a['bbox'][2],
				a['bbox'][1] + a['bbox'][3],
			]]], axis=0)
		return annotations