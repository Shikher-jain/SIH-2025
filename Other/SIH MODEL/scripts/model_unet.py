from tensorflow.keras import layers, models, Input

def unet_multioutput(input_shape=(512,512,21), filters=32, depth=4):
    inputs = Input(shape=input_shape)
    encs = []
    x = inputs

    for d in range(depth):
        f = filters * (2 ** d)
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        encs.append(x)
        x = layers.MaxPooling2D((2,2))(x)

    bn_filters = filters * (2 ** depth)
    x = layers.Conv2D(bn_filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(bn_filters, 3, activation='relu', padding='same')(x)

    for d in reversed(range(depth)):
        f = filters * (2 ** d)
        x = layers.Conv2DTranspose(f, 2, strides=(2,2), padding='same')(x)
        x = layers.concatenate([x, encs[d]])
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)
        x = layers.Conv2D(f, 3, activation='relu', padding='same')(x)

    heatmap = layers.Conv2D(1, 1, activation='sigmoid', name='heatmap')(x)

    y = layers.GlobalAveragePooling2D()(encs[-1])
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(1, name='yield')(y)

    return models.Model(inputs=inputs, outputs=[heatmap, y])
