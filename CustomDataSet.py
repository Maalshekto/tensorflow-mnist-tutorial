import numpy
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.datasets import base
import csv
import numpy as np

HWR_TRAINING = "train.csv"
HWR_TEST = "test.csv"


class CustomDataSet(object):
	def __init__(self, images, labels, fake_data=False, one_hot=False, dtype=dtypes.float32, reshape=True):
		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in (dtypes.uint8, dtypes.float32):
			raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
		assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
		self._num_examples = images.shape[0]

		# Convert shape from [num examples, rows, columns, depth]
		# to [num examples, rows*columns] (assuming depth == 1)
		if reshape:
			assert images.shape[3] == 1
			images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
		if dtype == dtypes.float32:
			# Convert from [0, 255] -> [0.0, 1.0].
			images = images.astype(numpy.float32)
			images = numpy.multiply(images, 1.0 / 255.0)
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
	
	@property
	def images(self):
		return self._images
	
	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

	def custom_load_csv_skip_header(filename, target_dtype, features_dtype, target_column=-1,  isTarget = True):
		with gfile.Open(filename) as csv_file:
			data_file = csv.reader(csv_file)
			data, target = [], []
			#skip first row.
			next(data_file)
			for row in data_file:
				if isTarget:
					y = row.pop(target_column)
					# Convert 0 to 9 factor to 10-bit array. Ex : 3 -> (0,0,0,1,0,0,0,0,0,0)
					label =  [0] * 10
					label[int(y)] = 1
					target.append(label)
				else:
					target.append([0] * 10)
				im = np.asarray(row, dtype=features_dtype)
				data.append(im.reshape(28, 28, 1))
		
		target = np.array(target, dtype=target_dtype)
		data = np.array(data)
		dict = {}
		dict['data'] = data
		dict['target'] = target
		return dict

def read_data_sets(filename, fake_data=False, one_hot=False, dtype=dtypes.float32,reshape=True, validation_size=5000):
	train = CustomDataSet.custom_load_csv_skip_header(filename=HWR_TRAINING, target_dtype=np.int, features_dtype=np.float32, target_column=0)
	test = CustomDataSet.custom_load_csv_skip_header(filename=HWR_TEST, target_dtype=np.int, features_dtype=np.float32, target_column=0,  isTarget = False)
		
	train_images = train['data']
	train_labels = train['target']
		
	test_images = test['data']
	test_labels = test['target']
	
	if not 0 <= validation_size <= len(train_images):
		raise ValueError('Validation size should be between 0 and {}. Received: {}.'.format(len(train_images), validation_size))

	validation_images = train_images[:validation_size]
	validation_labels = train_labels[:validation_size]
	train_images = train_images[validation_size:]
	train_labels = train_labels[validation_size:]
	
	test_images = test_images
	test_labels = test_labels

	train = CustomDataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
	validation = CustomDataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)
	test = CustomDataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
	return base.Datasets(train=train, validation=validation, test=test)

