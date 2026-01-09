# modules/model_cnn_lstm.py
from tensorflow.keras import layers, models, optimizers

def build_cnn_branch(input_shape_img):
    img_in = layers.Input(shape=input_shape_img, name="img_in")
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(img_in)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    return img_in, x

def build_lstm_branch(input_shape_seq):
    seq_in = layers.Input(shape=input_shape_seq, name="seq_in")
    y = layers.Masking()(seq_in)
    y = layers.LSTM(64, return_sequences=False)(y)
    y = layers.Dense(64, activation='relu')(y)
    return seq_in, y

def build_tabular_branch(input_shape_feats):
    feat_in = layers.Input(shape=input_shape_feats, name="feat_in")
    z = layers.Dense(64, activation='relu')(feat_in)
    z = layers.Dense(32, activation='relu')(z)
    return feat_in, z

def build_fusion_model(input_shape_img, input_shape_seq, input_shape_feats, n_classes=3):
    img_in, img_feat = build_cnn_branch(input_shape_img)
    seq_in, seq_feat = build_lstm_branch(input_shape_seq)
    feat_in, feat_feat = build_tabular_branch(input_shape_feats)

    combined = layers.concatenate([img_feat, seq_feat, feat_feat])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax', name='output')(x)

    model = models.Model(inputs=[img_in, seq_in, feat_in], outputs=out)
    # model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    sample_weight_mode="temporal"
)

    return model
