import numpy as np 
import tflearn

# Download the dataset
from tflearn.datasets import titanic
titanic.download_dataset('titanic_dataset.csv')

# Load CSV file and indicate first column representing labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0, categorical_labels=True, n_classes=2)

# Preprocessing function
def preprocess(data, columns_to_ignore):
	# Sort by descending id and delete columns
	for id in sorted(columns_to_ignore, reverse=True):
		[r.pop(id) for r in data]
	for i in range(len(data)):
		# Converting 'sex' field to float (id is 1 after removing labels column)
		data[i][1] = 1. if data[i][1] == 'female' else 0.
	return np.array(data, dtype=np.float32)


# Ignore 'name' and 'ticket' columns (id 1 and 6 of data array)
to_ignore = [1,6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradiant decent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)


# Testing the Model
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]

# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surving chances 
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surving Rate:", pred[0][1])
print("Winslet Surving Rate:", pred[1][1])


