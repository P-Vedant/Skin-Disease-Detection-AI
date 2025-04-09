import math
import functools
import gdown

def process_image(img, lbl, contrast_strength=1, tf=None):
  
    img=tf.cast(img, tf.float32)/255.0
    img=(tf.math.tanh((img-0.5)*contrast_strength)+1)/2 #use smooth contrast, as this is better for medical scans (varied lighting, critical mid-tones)
    img=tf.clip_by_value(img, 0.0, 1.0)

    return img, lbl

def collect_data(tf, gdown_file_id, train_path, test_path, image_scale, training_batch_size, testing_batch_size, contrast_strength):
    gdown.download_folder(f"https://drive.google.com/uc?if={gdown_folder_id}", output="Database")
        
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
