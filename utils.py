import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import smart_resize
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(
    __name__
)

def imshow(image: np.array, title: str):
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")

def prepare_dataset(dataset, target_size: tuple = (128, 128)) -> tuple:
    
    number_of_instances = tf.data.experimental.cardinality(dataset).numpy()
    
    x = np.zeros((number_of_instances, target_size[0], target_size[1], 3)).astype("float16")
    y = np.zeros((number_of_instances, target_size[0], target_size[1])).astype("uint8")
    
    for i, data in tqdm(enumerate(dataset)):
        logger.info(f"Original image shape: {data['image'].shape}")
        mask = data["segmentation_mask"].numpy().astype("uint8")
        
        resized_img = tf.image.resize(data["image"], size=target_size).numpy().astype("float16")
        logger.info(f"{(mask-1).shape}")
        mask = smart_resize(mask-1, size=target_size, interpolation="nearest").astype("uint8")
        
        x[i], y[i] = resized_img, mask[:,:,0]
    
    return x, y