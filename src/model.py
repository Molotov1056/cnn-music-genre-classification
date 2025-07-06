import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class GenreCNN:
    def __init__(self, input_shape=(128, 128, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        model = models.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_callbacks(self, model_path='models/best_model.h5'):
        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        if self.model is None:
            self.compile_model()
        
        callbacks = self.get_callbacks()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        return test_loss, test_accuracy
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        predictions = self.model.predict(X)
        return predictions
    
    def save_model(self, path):
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.model.save(path)
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        return self.model