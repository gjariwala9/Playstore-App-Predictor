from flask import Flask, render_template, session, url_for, redirect, flash
import numpy as np
from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, RadioField, IntegerField, SelectField
from wtforms.validators import DataRequired
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import joblib


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


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class ReviewForm(FlaskForm):
	review = TextField('Review', validators=[DataRequired()])

	submit = SubmitField("Predict")


class RatingForm(FlaskForm):
	
	size = TextField('Size', validators=[DataRequired()])

	size_type_lst = [
					('Mb', 'Mb'), 
					('Kb', 'Kb'), 
					('Varies with device', 'Varies with device'),
					]
	
	size_type = SelectField('Size Type', choices=size_type_lst, default=1)

	installs_lst = [
				('Installs 5+', 'Installs 5+'),
				('Installs 10+', 'Installs 10+'),
				('Installs 50+', 'Installs 50+'),
				('Installs 100+', 'Installs 100+'),
				('Installs 500+', 'Installs 500+'),
				('Installs 1,000+', 'Installs 1,000+'),
				('Installs 5,000+', 'Installs 5,000+'),
				('Installs 10,000+', 'Installs 10,000+'),
				('Installs 50,000+', 'Installs 50,000+'),
				('Installs 100,000+', 'Installs 100,000+'),
				('Installs 500,000+', 'Installs 500,000+'),
				('Installs 1,000,000+', 'Installs 1,000,000+'),
				('Installs 5,000,000+', 'Installs 5,000,000+'),
				('Installs 10,000,000+', 'Installs 10,000,000+'),
				('Installs 50,000,000+', 'Installs 50,000,000+'),
				('Installs 100,000,000+', 'Installs 100,000,000+'),
				('Installs 500,000,000+', 'Installs 500,000,000+'),
				('Installs 1,000,000,000+', 'Installs 1,000,000,000+')
                ]

	installs = SelectField('Number of Installs', choices=installs_lst, default=1)
	
	Type = RadioField('Type', coerce=float, default=0, choices=[(0.0,'Free'),(1.0,'Paid')])

	reviews = TextField('Number of Reviews', validators=[DataRequired()])
	
	submit = SubmitField("Predict")	

@app.route("/",methods=['GET','POST'])
def index():
	return render_template('index.html')


@app.route("/rating",methods=['GET','POST'])
def rating():

	form = RatingForm()

	if form.validate_on_submit():
		try:
			session['reviews'] = float(form.reviews.data)
			session['size'] = float(form.size.data)
			session['installs'] = form.installs.data
			session['Type'] = float(form.Type.data)
			session['size_type'] = form.size_type.data
		
			return redirect(url_for("rating_prediction"))

		except ValueError:
			flash('Invalid Input.', 'danger')

	
	return render_template('rating.html',form=form)

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = joblib.load('model_rf.pkl')

@app.route('/rating_prediction', methods=['GET'])
def rating_prediction():
	content = {}
	try:
		content['reviews'] = float(session['reviews'])
		content['size'] = float(session['size'])
		content['installs'] = str(session['installs'])
		content['Type'] = float(session['Type'])
		content['size_type'] = str(session['size_type'])

		results = return_rating_prediction(model, scaler, content)
	except KeyError:
		flash('Give some input first.', 'danger')
		return redirect(url_for("index"))

	return render_template('rating_prediction.html', results=results)



@app.route("/review",methods=['GET','POST'])
def review():

	form = ReviewForm()

	if form.validate_on_submit():
		try:
			session['review'] = str(form.review.data)
			return redirect(url_for("review_prediction"))

		except ValueError:
			flash('Invalid Input.', 'danger')

	
	return render_template('review.html',form=form)

cv = pickle.load(open('cv.pkl', 'rb'))
clf_model = joblib.load('lreg_clf_model.pkl')

@app.route('/review_prediction', methods=['GET'])
def review_prediction():
	content = {}
	try:
		content['review'] = str(session['review'])

		results = return_review_prediction(clf_model, cv, content)
	except KeyError:
		flash('Give some input first.', 'danger')
		return redirect(url_for("index"))

	return render_template('review_prediction.html', results=results)


@app.errorhandler(404)
def error_404(error):
	return render_template('errors/404.html'), 404

@app.errorhandler(403)
def error_403(error):
	return render_template('errors/403.html'), 403

@app.errorhandler(500)
def error_500(error):
	return render_template('errors/500.html'), 500


if __name__=='__main__':
	app.run(threaded=True, port=5000)	


