#!/bin/bash

# Verification script for uv setup
echo "🔍 Verifying uv setup..."

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Check uv version
echo "📋 uv version:"
uv --version

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "✅ Virtual environment (.venv) exists"
    echo "📦 Installed packages:"
    source .venv/bin/activate
    uv pip list | head -10
else
    echo "❌ Virtual environment (.venv) not found"
fi

echo "🎯 uv setup verification complete!"
