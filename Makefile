.PHONY: install dev test lint eval api ui docker

install:
	pip install -e ".[neural,ui,dev]"

dev:
	pip install -e ".[ui,dev]"

test:
	pytest -q

lint:
	ruff check . && ruff format --check .

eval:
	python -m kcopilot.evaluation --gate

api:
	uvicorn kcopilot.api:app --reload

ui:
	streamlit run ui/app.py

docker:
	docker compose up --build
