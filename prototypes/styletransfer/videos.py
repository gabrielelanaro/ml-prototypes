import cv2
import os
import numpy as np
import subprocess
from typing import List

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
