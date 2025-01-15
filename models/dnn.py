
def dnn(num_features, use_residual_blocks=True):
    # Input Layer
    input_layer = Input(shape=(num_features,))

    # Initial Dense Block
    x = Dense(256, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Residual Blocks
    if use_residual_blocks:
        for _ in range(5):
            block_input = x
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            # Adding skip connection
            x = Add()([x, block_input])

    # Dense Layers
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Output Layer
    output = Dense(1)(x)  # Regression output

    # Construct and compile model
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError(), root_mean_squared_error])

    return model