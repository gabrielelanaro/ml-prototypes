import cv2
import os
import numpy as np
import subprocess
from typing import List
from PIL import Image
import shutil

def fix_img(img: np.array) -> np.array:
    if len(img.shape) > 2 and img.shape[2] == 4:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def extract_frames_from_gif(gif_path: str) -> List[np.array]:
    # this assumes the gif has been already downloaded from S3 to EC2

    directory = "./frames"
    if os.path.exists("./frames"): shutil.rmtree(directory)
    os.makedirs(directory)

    subprocess.run(["ffmpeg", "-i",  f"{gif_path}", f"{directory}/frame%05d.png"])

    frames = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = Image.open(img_path)
        img = np.array(img)
        img = fix_img(img)
        frames.append(img)

    return frames

def make_video(images: List[np.array], 
               save_to: str, 
               video_name: str = "video.avi"):
  
  height, width, layers = images[0].shape

  video = cv2.VideoWriter(os.path.join(save_to, video_name), 0, 30, (width,height))

  for image in images:
      video.write(image)

  cv2.destroyAllWindows()
  video.release()
  
  convert_avi2mp4(os.path.join(save_to, video_name))
  
  return

def autocrop(image, threshold=10):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    
    return image
  
def add_topbottom_stripes(img):
  return cv2.copyMakeBorder(img, 40, 40, 0, 0, cv2.BORDER_CONSTANT, value=10)

def convert_avi2mp4(video_path):
  subprocess.run(["ffmpeg", "-y", "-i", f"{video_path}", f"{video_path.replace('avi', 'mp4')}"])
