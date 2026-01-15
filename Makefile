lint:
	uvx black src
	uvx isort --profile black src

test-embed:
	uv run src/color_perception_bench/embeddings.py

test-color:
	uv run src/color_perception_bench/colors.py

clean:
	find src -name "__pycache__" -exec rm -rf {} +
	find src -name "*.pyc" -exec rm -f {} +
