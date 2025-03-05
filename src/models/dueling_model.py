import tensorflow as tf
from tensorflow.keras import layers

def build_dueling_model(state_size, action_size, learning_rate=0.001):
    
    
    input_layer = layers.Input(shape=(state_size,))
    
    # Shared hidden layers with Batch Normalization
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Dueling network: Value and Advantage streams
    # Value stream
    value_fc = layers.Dense(128, activation='relu')(x)
    value = layers.Dense(1, activation='linear')(value_fc)
    
    # Advantage stream
    advantage_fc = layers.Dense(128, activation='relu')(x)
    advantage = layers.Dense(action_size, activation='linear')(advantage_fc)
    
    # Combine streams: Q = V + (A - mean(A))
    advantage_mean = layers.Lambda(lambda a: tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
    q_values = layers.Add()([value, layers.Subtract()([advantage, advantage_mean])])
    
    # Separate branch for continuous raise output
    raise_branch = layers.Dense(128, activation='relu')(x)
    raise_output = layers.Dense(1, activation='linear', name='raise_output')(raise_branch)
    
    # Create model with two outputs:
    # - 'action_output': Q-values for discrete actions
    # - 'raise_output': continuous raise amount
    model = tf.keras.Model(inputs=input_layer, outputs=[q_values, raise_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={'action_output': 'mse', 'raise_output': 'mse'}
    )
    
    return model
