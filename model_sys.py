def build_model(tf, conv_filters, conv_sizes, conv_stridesX, conv_stridesY, conv_LReLU_alphas, conv_pool_sizes, dense_sizes, dense_activation_methods):
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

def compile_model(tf, model):
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

def run_model(model, train_dataset, test_dataset, epochs):
    model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    return model.evaluate(test_dataset, verbose=0)
