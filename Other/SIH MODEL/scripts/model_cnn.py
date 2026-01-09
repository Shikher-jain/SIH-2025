from tensorflow.keras import layers, models

def build_model(input_shape=(512, 512, 21)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Yield output
    yield_out = layers.Dense(1, name='yield_output')(x)

    # Heatmap output (reshape to 512x512)
    x2 = layers.Dense(512 * 512, activation='sigmoid')(x)
    heatmap_out = layers.Reshape((512, 512, 1), name='heatmap_output')(x2)

    model = models.Model(inputs=inputs, outputs=[heatmap_out, yield_out])
    model.compile(optimizer='adam',
                  loss={'heatmap_output': 'binary_crossentropy', 'yield_output': 'mse'},
                  metrics={'yield_output': 'mae'})
    return model
