from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Conv2DTranspose, Concatenate, Input, Rescaling

encoder = Sequential(
    [
        Rescaling(1./255),
        Conv2D(64, 3, padding = "same", activation = "relu"),
        Conv2D(64, 3, padding = "same", activation = "relu"),
        MaxPool2D(2),
        Conv2D(128, 3, padding = "same", activation = "relu"),
        Conv2D(128, 3, padding = "same", activation = "relu"),
        MaxPool2D(2),
        Conv2D(256, 3, padding = "same", activation = "relu"),
        Conv2D(256, 3, padding = "same", activation = "relu"),
        MaxPool2D(2)
    ],
    name = "encoder"
)

decoder = Sequential(
    [
        Conv2DTranspose(256, 3, strides = 2, padding="same", activation = "relu"),
        Conv2D(256, 3, padding = "same", activation = "relu"),
        Conv2DTranspose(128, 3, strides = 2, padding="same", activation = "relu"),
        Conv2D(128, 3, padding = "same", activation = "relu"),
        Conv2DTranspose(64, 3, strides = 2, padding="same", activation = "relu"),
        Conv2D(64, 3, padding = "same", activation = "relu")
    ],
    name= "decoder"
)