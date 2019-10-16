import numpy as np

def seg_metric(res_label, gt_label, overlap_threshold=0.5):

	seg = 0.

	counter = 0
	imgCounter = 0

	assert(res_label == gt_label)
	assert(overlap_threshold >= 0.5)

	compare_dtype = np.dtype([('res', res_label.dtype), ('gt', gt_label.dtype)])

	sum_iou = 0
	sum_gt_objects = 0

	for t in range(len(res_label)):

		label_tuples = np.empty(res_label[t].shape, dtype=compare_dtype)
		label_tuples['res'] = res_label[t]
		label_tuples['gt'] = gt_label[t]

		both_foreground = np.logical_and(label_tuples['res'] > 0, label_tuples['gt'] > 0)
		index_pairs, intersections = np.unique(label_tuples[both_foreground], return_counts=True)
		gt_indexes, gt_size = np.unique(label_tuples['gt'][label_tuples['gt'] > 0], return_counts=True)
		sum_gt_objects += len(gt_indexes)

		for res_idx, gt_idx, intersection in zip(index_pairs, intersections):
			gt_size = (label_tuples['gt'] == gt_idx).sum()
			res_size = (label_tuples['res'] == res_idx).sum()
			overlap = intersection / gt_size
			if overlap > overlap_threshold:
				iou = intersection / (gt_size + res_size - intersection)
				sum_iou += iou

	if sum_gt_objects == 0:
		return 0.

	seg = sum_iou / sum_gt_objects
	return seg
