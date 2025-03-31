import os
import gdown
import shutil

def chk_repair_dataset(gdown_folder_id):
  if not (os.path.isdir("Database/Train") and os.path.isdir("Database/Test")):
    try:
      os.rmdir("Database")
    except:
      pass
    gdown.download_folder(f"https://drive.google.com/uc?if={gdown_folder_id}")
    if downloaded_folder and os.os.path.isdir(downloaded_folder):
      shutil.move(downloaded_folder, "Database")
