# Credits: Siraj

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import lightfmightFM

# Fetch data and format it
data = fetch_movielens(min_rating=4.0)

# Print training and test data
print(repr(data['train']))
print(repr(data['test']))

# Create model
model = LightFM(loss='warp')
# Train model
model.fit(data['train'], epochs=30, num_threads=2)

def recommendation(model, data, user_ids):

	# Number of users and movies in training data
	num_users, num_movies = data['train'].shape

	# Generate recommendatoins for each user we input
	for user in user_ids:
		# movies they already like
		known_positives = data['item_labels'][data['train'].tocsr()[user].indices]
		# Predicted movies from the model prediction
		scores = model.predict(user, np.arange(num_movies))
		# Rank most liked to least
		top_movies = data['item_labels'][np.argsort(-scores)]

		# Print out results
		print("User %s" % user)
		print("     Known positives:")

		for x in known_positives[:3]:
			print("        %s" % x)

		print("    Recommended:")
		for x in top_movies[:3]:
			print("        %s" %x)

recommendation(model, data, [5, 25, 500])