# 💊 Smart Pill Recognition - Quick Demo

## 🚀 Super Quick Start (3 steps)

```bash
# 1. Setup (one-time only)
sudo ./setup

# 2. Start the app  
./run

# 3. Open browser
# Go to: http://localhost:8501
```

## � Interactive Demo

```bash
# Run interactive demo with menu
./demo
```

## �🎯 Basic Usage

1. **Upload a pill image** (drag & drop)
2. **Enter any text on the pill** (optional)
3. **Click "Analyze"**
4. **View results** with confidence scores

## 🛠️ All Commands

| Command | What it does |
|---------|-------------|
| `./setup` | Install everything (drivers, CUDA, dependencies) |
| `./run` | Start the web app |
| `./test` | Check if everything works |
| `./deploy` | Deploy for production |
| `./monitor` | Watch GPU usage |
| `./clean` | Clean up temporary files |
| `./demo` | Interactive demo menu |

## 🔧 Options

```bash
# Different ways to start
./run                # Normal mode
./run --dev          # Development mode
./run --docker       # Docker mode
./run --port 8080    # Custom port

# Different tests
./test               # Quick test
./test --gpu         # Test GPU
./test --full        # Complete test

# Monitoring
./monitor            # Real-time GPU stats
./monitor --health   # System health check
```

## 🆘 Common Issues

**GPU not found?**
```bash
nvidia-smi           # Check if GPU is detected
sudo ./setup        # Reinstall drivers
```

**CUDA issues?**
```bash
./test --cuda        # Test CUDA
nvcc --version       # Check CUDA version
```

**App won't start?**
```bash
./test --app         # Test application
./run --debug        # Start with debug info
```

**Need help?**
```bash
./test --full        # Run all diagnostics
./monitor --health   # Check system health
```

---

*Made with ❤️ for pharmaceutical safety*
