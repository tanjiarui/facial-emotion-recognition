import config, cv2, os, pandas as pd, numpy as np
from tqdm.auto import tqdm
from glob import glob
from retinaface import RetinaFace

# anger/0594, anger/0508, anger/0500, anger/0667, anger/0616, anger/0806, disgust/0012, disgust/0051, disgust/0115, disgust/0193, disgust/0773
# image = cv2.imread('./train/disgust/0773.png')
# centroid = np.array([(image.shape[1] // 2, image.shape[0] // 2)]).reshape(-1)
# box, scores = list(), list()
# faces = RetinaFace.detect_faces(image, allow_upscaling=False)
# if not faces:
# 	print('skip the image')
# for face in faces.values():
# 	box.append(face['facial_area'])
# 	# confidence = face['score']
# 	area = (face['facial_area'][2] - face['facial_area'][0]) * (face['facial_area'][3] - face['facial_area'][1])
# 	center = [(face['facial_area'][0] + face['facial_area'][2]) // 2, (face['facial_area'][1] + face['facial_area'][3]) // 2]
# 	distance = np.linalg.norm(np.array(center) - centroid)
# 	print(area, distance, area / distance)
# 	scores.append(area / distance)
# best = scores.index(max(scores))
# x_min, y_min, x_max, y_max = box[best]
# cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
# cv2.imwrite('tmp.png', image)


def annotation(root):
	metadata = pd.DataFrame(columns=['file name', 'x', 'y', 'width', 'height'])
	for category in categories:
		images = glob(os.path.join(root, category, '*.png'))
		for file in tqdm(images):
			name = file.split('/')[3] + '/' + file.split('/')[4][:-4]
			image = cv2.imread(file)
			# centroid of the image
			centroid = np.array([(image.shape[1] // 2, image.shape[0] // 2)]).reshape(-1)
			box, scores = list(), list()
			# detect faces
			faces = RetinaFace.detect_faces(image, model=detector, allow_upscaling=False)
			if isinstance(faces, tuple):
				print('skip the image %s' % name)
				continue
			for face in faces.values():
				box.append(face['facial_area'])
				area = (face['facial_area'][2] - face['facial_area'][0]) * (face['facial_area'][3] - face['facial_area'][1])  # face area
				center = [(face['facial_area'][0] + face['facial_area'][2]) // 2, (face['facial_area'][1] + face['facial_area'][3]) // 2]  # center of the face
				distance = np.linalg.norm(np.array(center) - centroid)
				scores.append(area / distance)
			# the face close to the center most
			best = scores.index(max(scores))
			x_min, y_min, x_max, y_max = box[best]
			x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
			metadata = metadata.append({'file name': name, 'x': x, 'y': y, 'width': w, 'height': h}, ignore_index=True)
	return metadata


data_root = config.arguments.get('dataset')
train_root = os.path.join(data_root, 'train')
test_root = os.path.join(data_root, 'test')
categories = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
detector = RetinaFace.build_model()
data = annotation(train_root)
data.to_csv('bounding box train', index=False)
data = annotation(test_root)
data.to_csv('bounding box test', index=False)