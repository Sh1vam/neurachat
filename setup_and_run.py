#!/usr/bin/env python3
"""
NeuraChat Ultra — Setup & Launch Script
Run this once to install dependencies and start the server.

Usage:
    python setup_and_run.py          # Install deps + start server
    python setup_and_run.py --run    # Just start server (skip install)
"""

import subprocess, sys, os
from pathlib import Path

def run(cmd, check=True):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, check=False)
    if check and result.returncode != 0:
        print(f"  ⚠️  Command failed (returncode {result.returncode}) — continuing...")
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         NeuraChat Ultra v2.0 — Setup & Launch               ║
╚══════════════════════════════════════════════════════════════╝
    """)

    skip_install = '--run' in sys.argv

    if not skip_install:
        print("📦 Installing core dependencies...")
        run("pip install fastapi uvicorn[standard] python-multipart groq pydantic aiofiles python-dotenv httpx")
        run("pip install PyPDF2 python-docx Pillow tiktoken")
        run("pip install python-pptx reportlab nbformat matplotlib numpy")

        print("\n📦 Attempting optional dependencies (may fail if not compatible)...")
        run("pip install chromadb", check=False)
        run("pip install sentence-transformers", check=False)

        print("\n✅ Core installation complete!")
        print("   (Optional: chromadb and sentence-transformers for advanced RAG)")

    # Create .env file if GROQ_API_KEY not set
    if not os.getenv("GROQ_API_KEY"):
        env_file = Path(".env")
        if not env_file.exists():
            env_file.write_text('GROQ_API_KEY=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\n')
            print("\n✅ Created .env file with Groq API key")

    # Create outputs directory
    Path("outputs").mkdir(exist_ok=True)

    print("""
╔══════════════════════════════════════════════════════════════╗
║   Starting NeuraChat Ultra...                                ║
╠══════════════════════════════════════════════════════════════╣
║   Frontend:  http://localhost:8000                           ║
║   API Docs:  http://localhost:8000/docs                      ║
║   Health:    http://localhost:8000/api/health                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    os.execvp(sys.executable, [sys.executable, "-m", "uvicorn", "main:app",
                                "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    main()
