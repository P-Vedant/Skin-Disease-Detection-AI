import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import tensorflow as tf
import functools
import flask
import platform
import os

def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
def note(msg):
    input(msg+"\nPress enter to continue.\n")
    clear_screen()

def build_model(conv_filters, conv_sizes, conv_stridesX, conv_stridesY, conv_LReLU_negative_slopes, conv_pool_sizes, dense_sizes, dense_activation_methods):
    seq=[]

    for i in range(0, len(conv_filters)):
        seq.append(
            tf.keras.layers.Conv2D(
                filters=int(conv_filters[i]),
                kernel_size=(int(conv_sizes[i]),int(conv_sizes[i])),
                strides=(int(conv_stridesX[i]),int(conv_stridesY[i])),
                activation=tf.keras.layers.LeakyReLU(negative_slope=float(conv_LReLU_negative_slopes[i])),
                padding="same",
                use_bias=True
            )
        )

        seq.append(
            tf.keras.layers.MaxPooling2D((int(conv_pool_sizes[i]),int(conv_pool_sizes[i])))
        )
    
    seq.append(tf.keras.layers.Flatten())

    for i in range(0, len(dense_sizes)):
        seq.append(
            tf.keras.layers.Dense(int(dense_sizes[i]), activation=dense_activation_methods[i])
        )
    
    return tf.keras.models.Sequential(seq)

def compile_model(model):
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

def train_model(model, train_dataset, test_dataset, epochs):
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    return model.evaluate(test_dataset, verbose=0)

def process_image(img, lbl, contrast_strength=1):
  
    img=tf.cast(img, tf.float32)/255.0
    img=(tf.math.tanh((img-0.5)*contrast_strength)+1)/2 #use smooth contrast, as this is better for medical scans (varied lighting, critical mid-tones)
    img=tf.clip_by_value(img, 0.0, 1.0)

    return img, lbl

def collect_data(image_scale, training_batch_size, testing_batch_size, contrast_strength, shuffle_level):
    image_scale=int(image_scale)

    print("Collecting testing data...")
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "Database/test",
        image_size=(image_scale, image_scale),
        batch_size=int(testing_batch_size),
        label_mode='int'
    )

    print("Collecting training data...")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "Database/train",
        image_size=(image_scale, image_scale),
        batch_size=int(training_batch_size),
        label_mode='int'
    )

    print("Mapping data...")
    train_dataset=train_dataset.map(functools.partial(process_image, contrast_strength=float(contrast_strength)))
    test_dataset=test_dataset.map(functools.partial(process_image, contrast_strength=float(contrast_strength)))

    print("Configuring dataset objects pipeline modes...")
    train_dataset=train_dataset.shuffle(int(shuffle_level)).prefetch(tf.data.AUTOTUNE)
    test_dataset=test_dataset.prefetch(tf.data.AUTOTUNE)

    return (train_dataset, test_dataset)

def load_config():
    config_raw=None

    with open("config.txt","r") as f:
        config_raw=f.read()
    
    config_raw=config_raw.split("\n")
    
    config={}

    for line in config_raw:
        line=line.strip()
        line=line.split(":")
        config[line[0]]=line[1]
    
    return config

def init_AI(config):
  print("Building model...")
  model=build_model(
    config["mod.conv_filters"].split(","),
    config["mod.conv_sizes"].split(","),
    config["mod.conv_stridesX"].split(","),
    config["mod.conv_stridesY"].split(","),
    config["mod.conv_LReLU_negative_slopes"].split(","),
    config["mod.conv_pool_sizes"].split(","),
    config["mod.dense_sizes"].split(","),
    config["mod.dense_activation_methods"].split(",")
  )

  print("Compiling model...")
  compile_model(model)

  print("Collecting data...")
  train_dataset, test_dataset=collect_data(
    config["pre.image_scale"], 
    config["pre.train_batch_size"], 
    config["pre.test_batch_size"], 
    config["pre.contrast_strength"],
    config["pre.shuffle_level"]
  )
  
  return (model, train_dataset, test_dataset)

def init_flask(loaded_model):
  #add vedants code here
  #make sure to call the main loop somewhere on a seperate thread
  pass

def train_AI_UI(config):
    try:
        print("Initializing sequenctial AI model...")
        model, train_dataset, test_dataset=init_AI(config)
    except Exception as e:
        print("The AI model could not be built. This may be due to an invalid configuration parameter.")
        note(e)
        return None
    try:
        print("Training AI...")
        print(train_model(model, train_dataset, test_dataset, int(config["run.epochs"])))
    except Exception as e:
        print("The model could not be trained. This may be due to an invalid configuration parameter.")
        note(e)
        return None
    
    print("Saving model...")
    last_version=None
    with open("Models/last.txt","r") as f:
        last_version=f.read()
    with open("Models/last.txt","w") as f:
        f.write(last_version+1)
    model.save(f"Models/model_{last_version}.h5")

    note("The model is now trained and saved.")

def init():
    #load the config
    try:
        print("Loading configuration file...")
        config=load_config()
    except Exception as e:
        print("The configuration file could not be loaded. Please check for any errors.")
        note(e)
    
    clear_screen()

    loaded_model=None

    while True:
        #get the users option
        op=None
        while True:
            try:
                op=int(input("Please select an option:\n1. Load a model\n2. Train a model\n3. Open server\n>>> "))
                if op in (1,2,3):
                    break
                note("Please enter a number (key) in the range 1-2.")
            except:
                note("Please enter a number (key) in the range 1-2.")
        
        clear_screen()

        #run that option
        if op==1:
            #find the file to select, or leave as None if none could be located
            selection=None
            print("Searching for saved models...")
            if not os.path.isdir("Models"):
                print("No model directory found! Generating a directory...")
                os.makedirs("Models")
                with open("Models/last.txt","w") as f:
                    f.write("0")
            else:
                versions=None
                file_names=[]
                try:
                    with open("Models/last.txt","r") as f:
                        versions=f.read()
                    versions=int(versions)
                    while versions>=0:
                        file_name=f"Models/model_{versions}.h5"
                        if os.path.exists(file_name):
                            file_names.append(file_name)
                        versions-=1
                except:
                    print("Warning, the model max version number could not be determined.\nOverwritting content with 0...")
                    with open("Models/last.txt","w") as f:
                        f.write("0")
                    print("Using alternative searching method to scan continuous models (at most 1000)...")
                    versions=0
                    while versions<1000:
                        file_name=f"Models/model_{versions}.h5"
                        if os.path.exists(file_name):
                            file_names.append(file_name)
                        else:
                            break
                        versions+=1
                if len(file_names)==0:
                    print("No saved models found!")
                elif len(file_names)==1:
                    note(f"Exactly 1 saved model has been located.\nUsing `{file_names[0]}`.")
                    selection=file_names[0]
                else:
                    print(f"Located {len(file_names)} models. Please enter the number of the model you would like to use:")
                    for i in range(0,len(file_names)):
                        print(f"{i+1}. {file_names[i]}")
                    while True:
                        selection=int(input(">>> "))-1
                        if selection>=0 and selection<len(file_names):
                            break
                        else:
                            print(f"Please enter a number (key) in the range of 1 to {len(file_names)}:")
                    clear_screen()
            if not selection is None:
                selection=file_names[selection]
                #continue code here!!1
            else:
                note("Please download or train a model to select a loaded model.")
        elif op==2:
            train_AI_UI(config)
        elif op==3:
            if loaded_model is None:
                note("Please load a model before opening the server.")
            else:
                try:
                    init_flask(loaded_model)
                    note("Server closed.")
                except Exception as e:
                    print("An exception occured while operating the flask server, forcing it to close.")
                    note(e)



if __name__=="__main__":
    note("Welcome to the Hackathon Workshop booting menu. From here, you can train, save, and operate models.")
    init()
