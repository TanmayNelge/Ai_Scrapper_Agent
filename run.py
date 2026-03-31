#!/usr/bin/env python3
"""
Run the AI Scraper server.

Usage:
    python run.py
    python run.py --headless         # Run browser in headless mode
    python run.py --port 8080        # Custom port
    python run.py --same-llm         # Use same LLM for binary + extraction
"""
import argparse
import uvicorn
from backend.config import CONFIG


def main():
    parser = argparse.ArgumentParser(description="AI Scraper v2.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--headless", action="store_true",
                        help="Run browser in headless mode")
    parser.add_argument("--same-llm", action="store_true",
                        help="Use same LLM endpoint for binary and main model")
    parser.add_argument("--llm-port", type=int, default=1234,
                        help="LM Studio main model port")
    parser.add_argument("--binary-port", type=int, default=1235,
                        help="LM Studio binary model port (ignored if --same-llm)")
    parser.add_argument("--vision", action="store_true",
                        help="Enable vision/screenshots (only if model supports images)")
    args = parser.parse_args()

    # Apply CLI overrides to config
    CONFIG.browser.headless = args.headless
    CONFIG.llm.vision_enabled = args.vision

    CONFIG.llm.main_base_url = f"http://127.0.0.1:{args.llm_port}/v1"

    if args.same_llm:
        CONFIG.llm.binary_base_url = CONFIG.llm.main_base_url
        CONFIG.llm.binary_model = CONFIG.llm.main_model
        print(f"Using SAME LLM on port {args.llm_port} for all tasks")
    else:
        CONFIG.llm.binary_base_url = f"http://127.0.0.1:{args.binary_port}/v1"
        print(f"Main LLM: port {args.llm_port}")
        print(f"Binary LLM: port {args.binary_port}")

    print(f"Browser: {'Headless' if args.headless else 'Headful'}")
    print(f"Vision: {'ENABLED' if args.vision else 'DISABLED (text-only triage)'}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API docs: http://localhost:{args.port}/docs")
    print(f"Frontend: http://localhost:{args.port}")
    print()
    print("IMPORTANT: Make sure LM Studio has a model LOADED and server STARTED on port 1234!")

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
