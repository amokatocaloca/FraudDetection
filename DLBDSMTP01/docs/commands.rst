.PHONY: all data train evaluate serve

all: data train evaluate

data:
	python src/preprocess_data.py

train:
	dvc repro train

evaluate:
	dvc repro evaluate

serve:
	FLASK_APP=src/app.py flask run --host=0.0.0.0
