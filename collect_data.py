"""
NOT FINISHED

todo:
  -add kaggle API liason
done:
  -data normalizing and karas-ify-ing
"""

import gdown
import math
import os

def process_image(img, lbl):
    img=tf.cast(img, tf.float32)/255.0
    img=(tf.math.tanh((img-0.5)*5)+1)/2
    img=tf.clip_by_value(img, 0.0, 1.0)

    return img, lbl

def chk_repair_dataset():
  if not (os.isdir("Database/Train") and os.isdir("Database/Test")):
      try:
        os.rmdir("Database")
      except:
        pass
      gdown.download_folder(f"https://drive.google.com/uc?if={gdown_file_id}")

def collect_data(image_scale, training_batch_size, testing_batch_size):
    chk_repair_dataset()
        
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=(image_scale, image_scale),
        batch_size=training_batch_size,
        label_mode='int'
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=(image_scale, image_scale),
        batch_size=testing_batch_size,
        label_mode='int'
    )

    return (train_dataset.map(process_image), test_dataset.map(process_image)
