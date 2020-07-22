import os
from PIL import Image

def load_image(filename):
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize(required_size)
    pixels = np.asarray(img)
    pixels = (pixels - 127.5) / 127.5
    return pixels


def load_faces(directory, n_faces):
    faces = list()
    for filename in os.listdir(directory):
        pixels = load_image(directory + filename)
        faces.append(pixels)
        if len(faces) >= n_faces:
            break
    return np.asarray(faces)
