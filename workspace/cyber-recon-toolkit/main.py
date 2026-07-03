"""Main entry point for Cyber Recon Toolkit."""

import sys


def scan(target: str) -> None:
    print(f"Scanning target: {target}...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <target>")
        sys.exit(1)
    scan(sys.argv[1])
