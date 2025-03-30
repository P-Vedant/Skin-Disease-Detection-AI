import subprocess
import sys

from collect_data import chk_repair_dataset

def install_and_init():
  install_commands_windows=[
    "tensorflow",
    "gdown"
  ]
  chk_repair_dataset()
