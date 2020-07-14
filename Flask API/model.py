import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.data.path.append('./nltk_data/')

def return_review_prediction(model, cv, sample_json):
    user_review = sample_json['review']

    corpus = []

    review = re.sub('[^a-zA-Z]', ' ', user_review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

    X = cv.transform(corpus).toarray()
    
    classes = np.array(['Negative', 'Neutral', 'Positive'])
    
    class_ind = model.predict(X)[0]
    
    return classes[class_ind]

def cat_to_var(var, lst):
    inputs = []
    for i in lst:
        if i == var:
            inputs.append(1.0)
        else:
            inputs.append(0.0)
    return inputs

def return_rating_prediction(model, scaler, sample_json):
	inputs = []

	inputs.append(sample_json['reviews'])
	inputs.append(sample_json['size'])

	size_type_lst = [
					'Varies with device',
					'Mb',
					'Kb'
					]

	size_type = sample_json['size_type']

	inputs.extend(cat_to_var(size_type, size_type_lst))

	installs_lst = [
					'Installs 1,000+',
					'Installs 1,000,000+',
					'Installs 1,000,000,000+',
					'Installs 10+',
					'Installs 10,000+',
					'Installs 10,000,000+',
					'Installs 100+',
					'Installs 100,000+',
					'Installs 100,000,000+',
					'Installs 5+',
					'Installs 5,000+',
					'Installs 5,000,000+',
					'Installs 50+',
					'Installs 50,000+',
					'Installs 50,000,000+',
					'Installs 500+',
					'Installs 500,000+',
					'Installs 500,000,000+',
					]

	installs = sample_json['installs']
	inputs.extend(cat_to_var(installs, installs_lst))

	inputs.append(sample_json['Type'])

	app = [inputs]

	app = scaler.transform(app)

	predicted_rating = model.predict(app)[0]

	return round(predicted_rating, 1)