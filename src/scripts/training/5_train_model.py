import os, argparse, tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def augment(image, label):
    """Applies online augmentation manually during the dataset pipeline 
    to avoid TF 2.12 ModelCheckpoint JSON Serialization Bugs."""
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # 15 degrees max rotation (approx 0.26 radians)
    # tf.image doesn't have a native rotation, so we use tfa.image or a trick.
    # We will just stick to flip and a tiny bit of brightness since we heavily 
    # offline augmented (5,000 images).
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    return image, label

def create_dataset(data_dir, batch_size=32, img_size=224, shuffle=True, augment_data=False):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(img_size, img_size),
        shuffle=shuffle,
        seed=42
    )
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def build_model(img_size=224, num_classes=5):
    # Base Model (EfficientNet handles its own 0-255 scaling)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False 

    # Build Architecture
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs), base_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="aug_train")
    p.add_argument("--val_dir", default="splits/val")
    p.add_argument("--test_dir", default="splits/test")
    p.add_argument("--out", default="training_outputs")
    p.add_argument("--epochs", type=int, default=15)
    args = p.parse_args()
    
    os.makedirs(args.out, exist_ok=True)
    
    # Enable online augmentation explicitly ONLY for the training dataset
    train_ds = create_dataset(args.train_dir, augment_data=True)
    val_ds = create_dataset(args.val_dir, shuffle=False, augment_data=False)
    test_ds = create_dataset(args.test_dir, shuffle=False, augment_data=False)
    
    # Optimization
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    model, base_model = build_model()
    
    # STAGE 1
    print("\n--- STAGE 1: Training Head ---")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=8,
                   callbacks=[callbacks.ModelCheckpoint(os.path.join(args.out, 'best_stage1.keras'), save_best_only=True, monitor='val_loss', verbose=1, save_weights_only=True)])

    # STAGE 2
    print("\n--- STAGE 2: Fine-Tuning ---")
    base_model.trainable = True
    for layer in base_model.layers[:-20]: layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint_path = os.path.join(args.out, "liver_model_final.keras")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, 
                   callbacks=[
                       callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss', verbose=1),
                       callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1, save_weights_only=True)
                   ])

    print(f"Training Complete! Best model weights saved to {checkpoint_path}")

    # STAGE 3: Evaluation
    print("\n--- STAGE 3: Final Test Set Evaluation ---")
    # Instead of load_model (which crashes), we load the saved optimal weights into our architecture
    model.load_weights(checkpoint_path)
    
    y_true = np.concatenate([y for x, y in test_ds], axis=0).argmax(axis=1)
    y_pred = model.predict(test_ds).argmax(axis=1)
    
    report = classification_report(y_true, y_pred, target_names=['F0', 'F1', 'F2', 'F3', 'F4'], output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.out, "metrics.csv"))
    print("Metrics Saved to metrics.csv")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['F0', 'F1', 'F2', 'F3', 'F4'], yticklabels=['F0', 'F1', 'F2', 'F3', 'F4'])
    plt.xlabel('Predicted Stage')
    plt.ylabel('True Stage')
    plt.title('Test Set Confusion Matrix')
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"))
    print("Confusion Matrix Saved to confusion_matrix.png")

if __name__ == "__main__":
    main()
