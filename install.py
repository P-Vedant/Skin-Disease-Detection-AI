import subprocess
import sys

from Collector.chk_repair_dataset import chk_repair_dataset

def install_packages(packages):
  for package in packages:
    print(f"Installing {package}...", end="", flush=True)
    res=subprocess.run([sys.executable, "-m", "pip", "install", package])
    if res.return_code==0:
      print("Success!")
    else:
      print("Failure!")
      print(f"details:\n{res.stderr}")
      print(f"As {package} is integral to the functionality of this project, this error must be resolved before continuing in the installation process.")
      sys.exit(1)

def install_and_init(packages):
  print("Installing integral packages...")
  install_modules(packages)

  print("Downloading sorted kaggle dataset via gdown...")
  chk_repair_dataset()

  print("The install protocol is now complete!")
