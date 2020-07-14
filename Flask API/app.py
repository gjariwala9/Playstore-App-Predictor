from flask import Flask, render_template, session, url_for, redirect, flash
from form import ReviewForm, RatingForm
from model import return_review_prediction, return_rating_prediction
import pickle
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

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


