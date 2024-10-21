from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D, Concatenate, Input

def create_model(input_shape_text, input_shape_num):
    # Define the text input
    text_input = Input(shape=(input_shape_text,), name='text_input')
    x = Embedding(input_dim=5000, output_dim=128, input_length=input_shape_text)(text_input)
    x = LSTM(128, return_sequences=True)(x)
    x = GlobalAveragePooling1D()(x)
    
    # Define the numerical input
    num_input = Input(shape=(input_shape_num,), name='num_input')
    y = Dense(128, activation='relu')(num_input)
    
    # Concatenate the outputs
    combined = Concatenate()([x, y])
    z = Dense(128, activation='relu')(combined)
    z = Dense(1, activation='sigmoid')(z)
    
    # Create the model
    model = Model(inputs=[text_input, num_input], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model