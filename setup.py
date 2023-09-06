import os
import subprocess
import sys

def get_site_packages_in_venv(venv_path):
    # Run a Python script to print site-packages directory within the virtual environment
    python_executable = os.path.join(venv_path, 'bin', 'python')  # Adjust 'bin' to 'Scripts' on Windows
    command = [python_executable, '-c', 'import site; print(site.getsitepackages()[0])']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Error: Could not identify site-packages directory within virtual environment.")
        sys.exit(1)
        
    return result.stdout.strip()

def create_pth_file():
    virtual_env_path = os.environ.get('VIRTUAL_ENV')

    if virtual_env_path is None:
        print("Error: No active virtual environment detected.")
        sys.exit(1)

    site_packages_dir = get_site_packages_in_venv(virtual_env_path)

    if not os.path.exists(site_packages_dir):
        print(f"Error: No site-packages directory found at {site_packages_dir}")
        sys.exit(1)

    cwd = os.getcwd()
    pth_file_path = os.path.join(site_packages_dir, 'current_directory.pth')

    with open(pth_file_path, 'w') as f:
        f.write(cwd)

    print(f"Created .pth file at {pth_file_path}")

if __name__ == "__main__":
    create_pth_file()
