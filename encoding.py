import json, numpy as np, pandas as pd
from tqdm.auto import tqdm


class NPEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(NPEncoder, self).default(obj)


def jsonfy(data):
	data.insert(loc=5, column='target', value=None)
	data.loc[:, 'target'] = data['file name'].apply(lambda x: x.split('/')[0])

	image_ids = data['file name'].unique()
	image_dict = dict(zip(image_ids, range(len(image_ids))))
	json_dict = {'images': list(), 'type': 'instances', 'annotations': list(), 'categories': list()}
	# build the image list
	for image_id in image_ids:
		image = {'file_name': image_id + '.png', 'height': 400, 'width': 712, 'id': image_dict[image_id]}
		json_dict['images'].append(image)
	categories = [
		{'super category': 'face', 'id': 1, 'name': 'anger'},
		{'super category': 'face', 'id': 2, 'name': 'disgust'},
		{'super category': 'face', 'id': 3, 'name': 'fear'},
		{'super category': 'face', 'id': 4, 'name': 'happy'},
		{'super category': 'face', 'id': 5, 'name': 'neutral'},
		{'super category': 'face', 'id': 6, 'name': 'sad'},
		{'super category': 'face', 'id': 7, 'name': 'surprise'}
	]
	json_dict['categories'].extend(categories)
	category_map = {'anger': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7}
	# build the annotation list
	for idx, row in tqdm(data.iterrows()):
		image_id = image_dict[row['file name']]
		category = row['target']
		annotation = {
			'area': row['width'] * row['height'],
			'iscrowd': 0,
			'image_id': image_id,
			'bbox': [row['x'], row['y'], row['width'], row['height']],
			'category_id': category_map[category],
			'id': idx,
			'segmentation': None
		}
		json_dict['annotations'].append(annotation)
	return json_dict


train = pd.read_csv('bounding box train')
test = pd.read_csv('bounding box test')
result = jsonfy(train)
file = open('annotations train.json', 'w', encoding='utf-8')
json_str = json.dumps(result, cls=NPEncoder)
file.write(json_str)
file.close()
result = jsonfy(test)
file = open('annotations test.json', 'w', encoding='utf-8')
json_str = json.dumps(result, cls=NPEncoder)
file.write(json_str)
file.close()