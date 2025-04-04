"""
NOT FINISHED

todo:
  -add kaggle API liason
done:
  -data normalizing and karas-ify-ing
"""

import math
import functools

def process_image(img, lbl, contrast_strength=1, tf=None):
  
    img=tf.cast(img, tf.float32)/255.0
    img=(tf.math.tanh((img-0.5)*contrast_strength)+1)/2 #use smooth contrast, as this is better for medical scans (varied lighting, critical mid-tones)
    img=tf.clip_by_value(img, 0.0, 1.0)

    return img, lbl

def collect_data(tf, gdown_file_id, image_scale, training_batch_size, testing_batch_size, contrast_strength):
    chk_repair_dataset(gdown_file_id)
        
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

    return (train_dataset.map(functools.partial(process_image, contrast_strength=contrast_strength, tf=tf), test_dataset.map(functools.partial(process_image, contrast_strength=contrast_strength, tf=tf)))
