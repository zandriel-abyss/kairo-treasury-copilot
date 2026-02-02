#!/usr/bin/env python3
"""
Kairo Treasury Copilot - Deployment Script

Automated deployment and setup for the Kairo Treasury Optimization system.
Handles environment setup, dependency installation, and system validation.
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path
import argparse

class KairoDeployer:
    """Deployment manager for Kairo Treasury Copilot."""

    def __init__(self, project_root=None):
        """Initialize deployer with project root directory."""
        self.project_root = Path(project_root or Path(__file__).parent)
        self.venv_path = self.project_root / '.venv'
        self.requirements_file = self.project_root / 'requirements.txt'

    def check_system_requirements(self):
        """Check basic system requirements."""
        print("🔍 Checking system requirements...")

        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print(f"❌ Python {python_version.major}.{python_version.minor} detected. Python 3.8+ required.")
            return False

        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")

        # Check if pip is available
        try:
            subprocess.run([sys.executable, '-m', 'pip', '--version'],
                         capture_output=True, check=True)
            print("✅ pip is available")
        except subprocess.CalledProcessError:
            print("❌ pip is not available")
            return False

        return True

    def create_virtual_environment(self):
        """Create Python virtual environment."""
        print("🏗️ Creating virtual environment...")

        if self.venv_path.exists():
            print("⚠️ Virtual environment already exists. Removing...")
            shutil.rmtree(self.venv_path)

        try:
            subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)],
                         check=True)
            print(f"✅ Virtual environment created at {self.venv_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False

    def install_dependencies(self):
        """Install project dependencies."""
        print("📦 Installing dependencies...")

        pip_executable = self.venv_path / 'bin' / 'pip'
        if not pip_executable.exists():
            pip_executable = self.venv_path / 'Scripts' / 'pip'  # Windows

        if not pip_executable.exists():
            print("❌ Could not find pip in virtual environment")
            return False

        try:
            # Upgrade pip first
            subprocess.run([str(pip_executable), 'install', '--upgrade', 'pip'],
                         check=True, capture_output=True)

            # Install requirements
            subprocess.run([str(pip_executable), 'install', '-r', str(self.requirements_file)],
                         check=True)
            print("✅ All dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

    def validate_data_files(self):
        """Validate that required data files exist."""
        print("📊 Validating data files...")

        required_files = [
            'data/ap_data.csv',
            'data/ar_data.csv',
            'data/fx_rates.csv'
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
            else:
                # Check if file has content
                try:
                    with open(full_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) < 2:  # Header + at least one data row
                            print(f"⚠️ {file_path} appears to be empty or incomplete")
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")
                    missing_files.append(file_path)

        if missing_files:
            print(f"❌ Missing or invalid data files: {missing_files}")
            print("Please ensure all required CSV files are present in the data/ directory.")
            return False

        print("✅ All required data files are present and valid")
        return True

    def run_system_tests(self):
        """Run system integration tests."""
        print("🧪 Running system integration tests...")

        python_executable = self.venv_path / 'bin' / 'python'
        if not python_executable.exists():
            python_executable = self.venv_path / 'Scripts' / 'python'  # Windows

        try:
            result = subprocess.run([
                str(python_executable),
                'tests/test_system_integration.py'
            ], capture_output=True, text=True, cwd=self.project_root)

            if result.returncode == 0:
                print("✅ All system integration tests passed!")
                return True
            else:
                print("❌ System integration tests failed:")
                print(result.stdout)
                print(result.stderr)
                return False

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run tests: {e}")
            return False

    def create_startup_scripts(self):
        """Create convenient startup scripts."""
        print("📝 Creating startup scripts...")

        # CLI startup script
        cli_script = self.project_root / 'start_cli.sh'
        python_exe = self.venv_path / 'bin' / 'python'

        with open(cli_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Kairo Treasury Copilot - CLI Startup Script
cd "{self.project_root}"
"{python_exe}" interface/interface_cli.py
""")

        cli_script.chmod(0o755)

        # Dashboard startup script
        dashboard_script = self.project_root / 'start_dashboard.sh'

        with open(dashboard_script, 'w') as f:
            f.write(f"""#!/bin/bash
# Kairo Treasury Copilot - Dashboard Startup Script
cd "{self.project_root}"
"{python_exe}" -m streamlit run interface/dashboard.py --server.port 8501
""")

        dashboard_script.chmod(0o755)

        print("✅ Startup scripts created:")
        print(f"  • CLI: ./start_cli.sh")
        print(f"  • Dashboard: ./start_dashboard.sh")

    def run_demo(self):
        """Run a quick demo to verify everything works."""
        print("🎬 Running deployment verification demo...")

        python_executable = self.venv_path / 'bin' / 'python'

        try:
            # Test data loading
            result = subprocess.run([
                str(python_executable), '-c',
                """
from modules.data_ingest import get_data_loader
loader = get_data_loader()
ap = loader.load_ap_data()
ar = loader.load_ar_data()
fx = loader.load_fx_data()
print(f"✅ Data loaded: {len(ap)} AP, {len(ar)} AR, {len(fx)} FX records")
"""
            ], capture_output=True, text=True, cwd=self.project_root, check=True)

            print(result.stdout.strip())

            # Test basic prediction
            result = subprocess.run([
                str(python_executable), '-c',
                """
from modules.fx_model import create_fx_predictor
from modules.data_ingest import get_data_loader
from datetime import datetime
loader = get_data_loader()
fx_data = loader.load_fx_data()
predictor = create_fx_predictor(fx_data)
result = predictor.predict_fx_rate('USD/EUR', datetime.now(), days_ahead=7)
print(f"✅ FX prediction working: {result['predicted_rate']:.4f}")
"""
            ], capture_output=True, text=True, cwd=self.project_root, check=True)

            print(result.stdout.strip())

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Demo failed: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return False

    def deploy(self, skip_tests=False, skip_demo=False):
        """Run complete deployment process."""
        print("🚀 Starting Kairo Treasury Copilot Deployment")
        print("=" * 50)

        steps = [
            ("Check System Requirements", self.check_system_requirements),
            ("Create Virtual Environment", self.create_virtual_environment),
            ("Install Dependencies", self.install_dependencies),
            ("Validate Data Files", self.validate_data_files),
        ]

        if not skip_tests:
            steps.append(("Run System Tests", self.run_system_tests))

        steps.extend([
            ("Create Startup Scripts", self.create_startup_scripts),
        ])

        if not skip_demo:
            steps.append(("Run Verification Demo", self.run_demo))

        success_count = 0

        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            if step_func():
                success_count += 1
            else:
                print(f"❌ {step_name} failed. Aborting deployment.")
                return False

        print("\n" + "=" * 50)
        if success_count == len(steps):
            print("🎉 DEPLOYMENT SUCCESSFUL!")
            print("\n📋 Next Steps:")
            print("1. Start CLI: ./start_cli.sh")
            print("2. Start Dashboard: ./start_dashboard.sh")
            print("3. Run tests anytime: python -m pytest tests/")
            print("\n🎯 Kairo Treasury Copilot is ready for use!")
            return True
        else:
            print(f"❌ DEPLOYMENT FAILED: {success_count}/{len(steps)} steps completed")
            return False

def main():
    """Main deployment entry point."""
    parser = argparse.ArgumentParser(description="Deploy Kairo Treasury Copilot")
    parser.add_argument('--project-root', help='Project root directory')
    parser.add_argument('--skip-tests', action='store_true', help='Skip system tests')
    parser.add_argument('--skip-demo', action='store_true', help='Skip verification demo')

    args = parser.parse_args()

    deployer = KairoDeployer(args.project_root)
    success = deployer.deploy(skip_tests=args.skip_tests, skip_demo=args.skip_demo)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()