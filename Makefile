modal:
	python -m modal setup

dev:
	modal deploy --env main service/api/main.py

run:
	modal run --env main service/backend/main.py

shell:
	modal shell --env main service/api/main.py
