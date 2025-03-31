from Collector.chk_repair_dataset import chk_repair_dataset

import subprocess
import sys

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

def init_env(packages, gdown_folder_id):
    print("Installing integral packages...")
    install_modules(packages)
    
    print("Downloading sorted kaggle dataset via gdown...")
    chk_repair_dataset(gdown_folder_id)

def load_config():
    config_raw=None

    with open("ProgramData/config.txt","r") as f:
        config_raw=f.read()
    
    config_raw=config_raw.split("\n")
    
    config={}

    for line in config_raw:
        line=line.strip()
        line=line.split(":")
        config[line[0]]=line[1]
    
    return config

def init_AI(config):
  from model_sys import build_model, compile_model
  from Collector.collect_data import collect_data
  
  print("Initializing OS environs...")
  
  import os
  os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
  
  print("Initializing tensorflow...")
  import tensorflow as tf
  
  print("Building model...")
  model=build_model(
    tf,
    config["mod.conv_filters"].split(","),
    config["mod.sizes"].split(","),
    config["mod.conv_stridesX"].split(","),
    config["mod.conv_stridesY"].split(","),
    config["mod.conv_LReLU_alpha"].split(","),
    config["mod.conv_pool_sizes"].split(","),
    config["mod.dense_sizes"].split(","),
    config["mod.dense_activation_methods"].split(",")
  )

  print("Compiling model...")
  compile_model(tf, model)

  print("Collecting data...")
  train_dataset, test_dataset=collect_data(tf, config["pre.gdown_file_id"], config["pre.train_batch_size"], config["pre.test_batch_size"], config["pre.contrast_strength"])
  
  return (config, model, train_dataset, test_dataset)

def chk_install_status():
    status=None
    with open("ProgramData/did_install","r") as f:
        status=f.read()
    return status

def init():
    print("Loading configuration file (ProgramData/config.txt)...")
    config=load_config()
    
    print("Checking installation status...")
    status=chk_install_status()

    if status!="1":
        print("Only minimal environemnt detected!\nStarting environmental initialization protocol...")
        init_env(config["pre.gdown_folder_id"]) 
        print("Environemntal initialization protcol complete!")
    print("Starting A.I. initialization protocol...")
    model_env=init_ai(config)
    print("Test INIT process V1.0.0 completed!")

if __name__=="__main__":
    init()
