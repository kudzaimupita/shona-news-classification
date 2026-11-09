.DEFAULT_GOAL := help
RUN := uv run

.PHONY: help install lint format typecheck test check baselines encoders llm-submit llm-collect summary

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install dependencies from the lockfile
	uv sync --frozen

lint: ## Lint with ruff
	$(RUN) ruff check src tests

format: ## Format with ruff
	$(RUN) ruff format src tests

typecheck: ## Type-check with mypy
	$(RUN) mypy

test: ## Run the test suite
	$(RUN) pytest

check: lint typecheck test ## Lint, type-check and test

baselines: ## Data summary, tokenizer fertility, majority and TF-IDF baselines
	$(RUN) shona-news data-summary
	$(RUN) shona-news fertility
	$(RUN) shona-news classical

encoders: ## Fine-tune every configured encoder run not yet recorded
	PYTORCH_ENABLE_MPS_FALLBACK=1 $(RUN) shona-news finetune-grid

llm-submit: ## Submit the language-model batch (asks for confirmation)
	$(RUN) shona-news llm-submit

llm-collect: ## Wait for the batch and record language-model runs
	$(RUN) shona-news llm-collect

summary: ## Aggregate recorded runs into results/summary.md
	$(RUN) shona-news summary
