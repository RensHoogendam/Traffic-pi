#!/bin/bash
# Traffic-pi Easy Setup Script

set -e  # Exit on error

echo "🚦 Traffic-pi Setup"
echo "===================="
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Install package in development mode
echo "📥 Installing Traffic-pi and dependencies..."
pip install -e . -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run a test: python test_system.py"
echo "  3. Try detection: traffic-pi --image path/to/image.jpg"
echo ""
echo "For help: traffic-pi --help"
