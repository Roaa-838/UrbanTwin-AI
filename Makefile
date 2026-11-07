.PHONY: install test demo eo roads geo traffic dashboard report
install:
	python -m pip install -e '.[all]'
test:
	python -m pytest -q
demo:
	urbantwin demo
eo:
	urbantwin eo-fetch
roads:
	urbantwin roads
geo:
	urbantwin geo-train
traffic:
	urbantwin traffic-run
dashboard:
	streamlit run src/urbantwin/visualization/dashboard.py
report:
	urbantwin report
