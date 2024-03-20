import os
import csv
from tkinter import Tk, Button, Label, PhotoImage
from PIL import Image, ImageTk, ImageDraw, ImageFont

class ImagePairViewer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.images_data = self.load_csv_data()
        self.current_index = 0
        self.coord_conversion = False
        self.flip = True
        self.rotate = 0
        self.wrong_augmented_coords = 0

        self.root = Tk()
        self.root.title("Image Viewer")
    
        # Bind keyboard shortcuts
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("<Left>", lambda event: self.prev_image())
                       
        self.setup_ui()
        self.display_current_image()

    def load_csv_data(self):
        data = {}
        with open(self.csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                _, img_path, label, x1, y1,_,_, x2, y2 = row[:9]
                box = (float(x1), float(y1), float(x2), float(y2))

                if 'aug_' not in img_path:
                    # Check if the original image path is already in the data dictionary
                    if img_path not in data:
                        aug_img_path = img_path.replace('/allImgs_extracted_smaller/', '/allImgs_extracted_smaller_aug/aug_')
                        data[img_path] = {'original': [], 'augmented': [], 'original_labels': [], 'augmented_labels': [], 'augmented_path': aug_img_path}

                    data[img_path]['original'].append(box)
                    data[img_path]['original_labels'].append(label)

                else:
                    # For augmented images, replace 'aug_' to find the corresponding original image entry
                    original_img_path = img_path.replace('aug_', '').replace('/allImgs_extracted_smaller_aug/', '/allImgs_extracted_smaller/')
                    if original_img_path not in data:
                        data[original_img_path] = {'original': [], 'augmented': [], 'original_labels': [], 'augmented_labels': [], 'augmented_path': img_path}

                    data[original_img_path]['augmented'].append(box)
                    data[original_img_path]['augmented_labels'].append(label)
        return data

    def setup_ui(self):
        
        self.fault_coords_label = Label(self.root, text=f"Photos with faulty coords: {self.wrong_augmented_coords}")
        self.fault_coords_label.pack(side="top")
        
        self.my_label = Label(self.root, text=f"{self.current_index+1}/{len(self.images_data)}")
        self.my_label.pack(side="top")
        
        self.original_image_label = Label(self.root)
        self.original_image_label.pack(side="left")

        self.augmented_image_label = Label(self.root)
        self.augmented_image_label.pack(side="right")

        self.rotate_button = Button(self.root, text=f"Rotate ({self.rotate})", command=self.toggle_rotate)
        self.rotate_button.pack(side="bottom")
        
        self.coord_button = Button(self.root, text="Converted Coords", command=self.toggle_convert)
        self.coord_button.pack(side="bottom")

        self.flip_button = Button(self.root, text="Non-Flipped Coords", command=self.toggle_flip)
        self.flip_button.pack(side="bottom")        
        
        self.next_button = Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side="bottom")

        self.prev_button = Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side="bottom")


    def display_current_image(self):
        paths = list(self.images_data.keys())
        # paths = sorted(self.images_data.keys(), key=lambda x: len(self.images_data[x]['original']) + len(self.images_data[x]['augmented']))
        if self.current_index >= len(paths):
            return

        img_path = paths[self.current_index]
        aug_path = self.images_data[img_path]['augmented_path']
        original_boxes = self.images_data[img_path]['original']
        augmented_boxes = self.images_data[img_path]['augmented']
        original_labels = self.images_data[img_path]['original_labels']  # Assuming label names are stored here
        augmented_labels = self.images_data[img_path]['augmented_labels']

        orig = Image.open(img_path).rotate(self.rotate)
        aug = Image.open(aug_path).rotate(self.rotate)
        original_img = self.draw_boxes(orig, original_boxes, True, img_path, original_labels)
        try:
            augmented_img = self.draw_boxes(aug, augmented_boxes, False, aug_path, augmented_labels)
            augmented_photo = ImageTk.PhotoImage(augmented_img)
        except Exception as e:
            print(f"Error loading augmented image: {e}")
            print(aug_path)
            print(augmented_boxes)
            self.wrong_augmented_coords += 1
            self.fault_coords_label.config(text=f"Photos with faulty coords: {self.wrong_augmented_coords}")
            # Option 1: Clear the augmented image label
            augmented_photo = PhotoImage()    

        original_photo = ImageTk.PhotoImage(original_img)
        

        self.original_image_label.config(image=original_photo)
        self.original_image_label.image = original_photo  # Keep a reference!

        self.augmented_image_label.config(image=augmented_photo)
        self.augmented_image_label.image = augmented_photo  # Keep a reference!


    def draw_boxes(self, img, boxes, orig, img_path, labels):
        draw = ImageDraw.Draw(img)
        img_width, img_height = img.size
        font_size = 20  # Set this to your preferred font size
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        for box, label in zip(boxes, labels):
            # Assuming box coordinates are normalized (i.e., in the range [0, 1])
            if self.coord_conversion:
                if self.flip:
                    scaled_box = (
                        box[1] * img_height,  # Scale x1
                        (1- box[2]) * img_width, # Scale y1
                        box[3] * img_height,  # Scale x2
                        (1- box[0]) * img_width  # Scale y2
                    )
                else:
                    scaled_box = (
                        box[1] * img_width,  # Scale x1
                        (1- box[2]) * img_height, # Scale y1
                        box[3] * img_width,  # Scale x2
                        (1- box[0]) * img_height  # Scale y2
                    )                     
            else:
                if self.flip:
                    scaled_box = (
                        box[0] * img_width,  # Scale x1
                        box[1] * img_height, # Scale y1
                        box[2] * img_width,  # Scale x2
                        box[3] * img_height  # Scale y2
                    )                 
                else:
                    scaled_box = (
                        box[0] * img_height,  # Scale x1
                        box[1] * img_width, # Scale y1
                        box[2] * img_height,  # Scale x2
                        box[3] * img_width,  # Scale y2
                    )    
            draw.rectangle(scaled_box, outline="red", width=3)  # Draw the box
            draw.text((scaled_box[0], scaled_box[1]), label, font=font, fill="black")  # Draw the label
        return img


    def next_image(self):
        self.current_index += 1
        self.my_label.config(text=f"{self.current_index+1}/{len(self.images_data)}")
        self.display_current_image()

    def prev_image(self):
        self.current_index = max(0, self.current_index - 1)
        self.my_label.config(text=f"{self.current_index+1}/{len(self.images_data)}")
        self.display_current_image()
        
    def toggle_rotate(self):
        self.rotate = (self.rotate+90) % 360  
        self.rotate_button.config(text=f"Rotate ({self.rotate})")
        self.display_current_image()
        
    def toggle_convert(self):
        self.coord_conversion = not self.coord_conversion
        if self.coord_conversion:
            self.coord_button.config(text = "Converted Coords")
        else:
            self.coord_button.config(text = "Normal Coords")
        self.display_current_image()
        
    def toggle_flip(self):
        self.flip = not self.flip
        if self.flip:
            self.flip_button.config(text = "Flipped Coords")
        else:
            self.flip_button.config(text = "Non-Flipped Coords")
        self.display_current_image()

if __name__ == "__main__":
    csv_file = "/home/alex/tflite_model_maker_wsl2/annotations/cross_val/aug_5707_cv_fold_0_no_other.csv"
    viewer = ImagePairViewer(csv_file)
    viewer.root.mainloop()
