import os  
import cv2  
  
class PhotoExtractor():  
    def __init__(self, filepath, frame_number):  
        self.filepath = filepath  
        self.frame_number = frame_number  
  
    def extract_photos(self):  
        photos = []  
        for filename in os.listdir(self.filepath):  
            if filename.startswith("view") and len(filename) == 8:  
                camera_number = int(filename[4:6])  
                frame_num = int(filename[8:10])  
                if frame_num == self.frame_number:  
                    file_path = os.path.join(self.filepath, filename)  
                    img = cv2.imread(file_path)  
                    photos.append(img)  
        return photos  