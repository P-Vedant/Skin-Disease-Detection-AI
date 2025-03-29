import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import tensorflow as tf

def build_model(conv_filters, conv_sizes, conv_stridesX, conv_stridesY, conv_LReLU_alphas, conv_pool_sizes, dense_sizes, dense_activation_methods):
    seq=[]


    for i in range(0, len(conv_filters)):
        seq.append(
            tf.keras.layers.Conv2D(
                filters=int(conv_filters[i]),
                kernel_size=(int(conv_sizes[i]),int(conv_sizes[i])),
                strides=(int(conv_stridesX[i]),int(conv_stridesY[i])),
                activation=tf.keras.layers.LeakyReLU(alpha=float(conv_LReLU_alphas[i]))
            )
        )

        seq.append(
            tf.keras.layers.MaxPooling2D((int(conv_pool_sizes[i]),int(conv_pool_sizes[i])))
        )
    
    seq.append(tf.keras.layers.Flatten())

    for i in range(0, len(dense_sizes)):
        seq.append(
            tf.keras.layers.Dense(dense_sizes[i], activation=dense_activation_methods[i])
        )
    
    return tf.keras.models.Sequential(seq)

def compile_model(model):
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

def run_model(model, train_dataset, test_dataset, epochs):
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    return model.evaluate(test_dataset, verbose=0)

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

def init():
    config=load_config()

    model=build_model(
        config["mod.conv_filters"].split(","),
        config["mod.sizes"].split(","),
        config["mod.conv_stridesX"].split(","),
        config["mod.conv_stridesY"].split(","),
        config["mod.conv_LReLU_alpha"].split(","),
        config["mod.conv_pool_sizes"].split(","),
        config["mod.dense_sizes"].split(","),
        config["mod.dense_activation_methods"].split(",")
    )

    compile_model(model)

    train_dataset, test_dataset=collect_data(config["pre.train_path"],config["pre.test_path"])

    train_dataset.map(process_image)
    test_dataset.map(process_image)

    return (config, model, train_dataset, test_dataset)

def main():
    print("Initializing local side...")
    init_data=init()
    print("Local side initiated!")

if __name__=="__main__":
    main()
