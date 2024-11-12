from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from training.preprocess import process_dataset

train_data, train_labels, test_data, test_labels, max_frames = process_dataset()

# shape(n, 102, 24, 3)

pose_points_count = 24
pose_point_features_count = 3
actions_count = 3

input_shape = (max_frames, pose_points_count, pose_point_features_count)
input_layer = layers.Input(shape=input_shape)

cnn_layer = layers.TimeDistributed(layers.Conv1D(32, kernel_size=3, activation='relu'))(input_layer)
cnn_layer = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(cnn_layer)
cnn_layer = layers.TimeDistributed(layers.Conv1D(64, kernel_size=3, activation='relu'))(cnn_layer)
cnn_layer = layers.TimeDistributed(layers.MaxPooling1D(pool_size=2))(cnn_layer)

cnn_layer = layers.TimeDistributed(layers.Flatten())(cnn_layer)
lstm_layer = layers.LSTM(128, return_sequences=False)(cnn_layer)

dense_layer = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm_layer)
dense_layer = layers.Dropout(0.6)(dense_layer)
output_layer = layers.Dense(actions_count, activation='softmax')(dense_layer)

model = models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

history = model.fit(
    train_data,
    train_labels,
    validation_data=(test_data, test_labels),
    epochs=20,
    batch_size=32,
    callbacks = [early_stopping, lr_scheduler]
)

model.save('pose_cnn_lstm_model.keras')

