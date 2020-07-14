from flask_wtf import FlaskForm
from wtforms import TextField, SubmitField, RadioField, IntegerField, SelectField
from wtforms.validators import DataRequired

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