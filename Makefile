# Agentic RAG for Financial Statement & Wealth Intelligence
.PHONY: help install data index eval eval-full test serve demo clean rebuild

help:
	@echo "make install   install dependencies"
	@echo "make data      fetch SEC filings and build the corpus + portfolio"
	@echo "make index     build the hybrid Qdrant index (slow: embeds 1.3k chunks)"
	@echo "make eval      retrieval metrics: Recall@10, constraint compliance"
	@echo "make eval-full evaluation with the ablation grid"
	@echo "make test      run the test suite"
	@echo "make serve     start the API on :8000"
	@echo "make demo      scripted walkthrough with full traces"
	@echo "make rebuild   data + index from scratch"

install:
	python -m pip install -r requirements.txt

data:
	python -m scripts.fetch_sec
	python -m scripts.build_corpus
	python -m scripts.build_portfolio
	python -m scripts.build_eval_set

index:
	python -m scripts.index_corpus

eval:
	python -m scripts.evaluate

eval-full:
	python -m scripts.evaluate --ablation --json data/eval/results.json

test:
	python -m pytest tests/ -q

serve:
	python -m uvicorn app.main:app --reload --port 8000

demo:
	python -m scripts.demo

rebuild: data index

clean:
	rm -rf data/qdrant .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
