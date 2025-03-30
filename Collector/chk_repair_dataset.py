def chk_repair_dataset(gdown_file_id):
  if not (os.isdir("Database/Train") and os.isdir("Database/Test")):
      try:
        os.rmdir("Database")
      except:
        pass
      gdown.download_folder(f"https://drive.google.com/uc?if={gdown_file_id}")
