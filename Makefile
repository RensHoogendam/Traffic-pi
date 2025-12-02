.PHONY: help setup install test clean run-test run-camera lint format

help:
	@echo "Traffic-pi Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup     - Complete setup (venv + install)"
	@echo "  make install   - Install dependencies only"
	@echo ""
	@echo "Testing:"
	@echo "  make test      - Run test suite"
	@echo "  make run-test  - Run system test"
	@echo ""
	@echo "Running:"
	@echo "  make run-camera - Start camera detection"
	@echo ""
	@echo "Development:"
	@echo "  make format    - Format code with black"
	@echo "  make lint      - Run linters"
	@echo "  make clean     - Remove build artifacts"
	@echo ""

setup:
	@chmod +x setup.sh
	@./setup.sh

install:
	@echo "📥 Installing dependencies..."
	@pip install -e . -q
	@echo "✅ Installation complete"

test:
	@echo "🧪 Running tests..."
	@pytest tests/ -v

run-test:
	@echo "🚦 Running system test..."
	@python test_system.py

run-camera:
	@echo "📹 Starting camera detection (Press 'q' to quit)..."
	@traffic-pi --camera

lint:
	@echo "🔍 Running linters..."
	@flake8 src/ --max-line-length=100 || echo "⚠️  Install flake8: pip install flake8"

format:
	@echo "✨ Formatting code..."
	@black src/ || echo "⚠️  Install black: pip install black"

clean:
	@echo "🧹 Cleaning up..."
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf .pytest_cache
	@rm -rf **/__pycache__
	@rm -rf **/*.pyc
	@echo "✅ Clean complete"
