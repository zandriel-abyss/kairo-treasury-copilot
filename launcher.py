#!/usr/bin/env python3
"""
Kairo Treasury Copilot Launcher

Simple entrypoint that lets a user choose between:
- Running the CLI interface
- Launching the Streamlit dashboard

This file is designed to be the main target for PyInstaller when
building a standalone executable on Windows.
"""

import subprocess
import sys
import os


def run_cli() -> None:
    """Run the Kairo CLI."""
    # Ensure we run from the project root so relative paths work
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    from interface.interface_cli import main as cli_main

    cli_main()


def run_dashboard() -> None:
    """Run the Streamlit dashboard."""
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Use the current Python environment's Streamlit
    dashboard_path = os.path.join(project_root, "interface", "dashboard.py")

    # On Windows, PyInstaller bundles an embedded Python; this subprocess
    # call will use the embedded interpreter's environment.
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", dashboard_path],
        cwd=project_root,
        check=False,
    )


def main() -> None:
    """Simple text menu to choose interface."""
    while True:
        print("\n🎯 Kairo Treasury Copilot Launcher")
        print("1. Launch CLI")
        print("2. Launch Dashboard")
        print("3. Exit")

        choice = input("Enter choice (1-3): ").strip()

        if choice == "1":
            run_cli()
        elif choice == "2":
            run_dashboard()
        elif choice == "3":
            print("👋 Goodbye.")
            break
        else:
            print("Invalid choice, please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()

