# Quick Start Guide

## One-Command Setup

### Option 1: Using setup script (Recommended)

```bash
bash setup.sh
```

### Option 2: Using Make

```bash
make setup
```

That's it! The script will:

- Create a virtual environment
- Install all dependencies
- Set up the project in development mode

## After Setup

Activate the virtual environment:

```bash
source venv/bin/activate
```

## Quick Commands

### Using Make (Easy)

```bash
make help          # Show all available commands
make run-test      # Test the system
make run-camera    # Start camera detection
```

### Using CLI Directly

```bash
traffic-pi --help                           # Show help
traffic-pi --image data/test.jpg            # Detect in image
traffic-pi --video video.mp4                # Process video
traffic-pi --camera                         # Use camera
```

## Development

```bash
make test          # Run tests
make format        # Format code
make lint          # Check code quality
make clean         # Remove build files
```
