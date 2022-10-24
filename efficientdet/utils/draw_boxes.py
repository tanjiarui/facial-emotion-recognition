import cv2


def draw_boxes(image, boxes, scores, labels, colors, classes):
	for b, l, s in zip(boxes, labels, scores):
		class_id = int(l) + 1
		class_name = classes[class_id]
		x_min, y_min, x_max, y_max = list(map(int, b))
		score = '{:.4f}'.format(s)
		color = colors[class_id]
		label = '-'.join([class_name, score])

		ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
		cv2.rectangle(image, (x_min, y_max - ret[1] - baseline), (x_min + ret[0], y_max), color, -1)
		cv2.putText(image, label, (x_min, y_max - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)