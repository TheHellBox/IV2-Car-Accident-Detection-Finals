import os
from os.path import abspath, join
from PIL import Image
import cv2

def images_to_video(img_folder: str, video_name: str) -> cv2.VideoWriter:
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(img_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(img_folder, image)))

    cv2.destroyAllWindows()
    return video

class BestLogger3000:
    filepath = "."
    
    def __init__(self, filepath: str = "."):
        self.filepath = abspath(filepath)

    def save_text(self, fname: str, data: str):
        with open(join(self.filepath, fname), 'a+') as f:
            f.write(data)

    def save_image(self, fname: str, img: Image):
        return img.save(join(self.filepath, fname))

    def save_video(self, fname: str, imgf: str):
        images_to_video(fname, imgf).release()
