.PHONY: install pipeline quick app test
install:
	pip install -r requirements.txt && pip install -e .
pipeline:        ## full scale: 100k accounts, 1.6M transactions
	python -m fincrime.pipeline --scale full
quick:           ## 20k accounts, about a minute
	python -m fincrime.pipeline --scale medium
app:
	streamlit run app/Home.py
test:
	pytest -q
