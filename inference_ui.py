import os
import tkinter as tk
from PIL import Image, ImageTk
from pathlib import Path

def load_images(main_dir):
    images = {}
    model_names = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    for file in os.listdir(main_dir):
        if os.path.isfile(os.path.join(main_dir, file)):
            file_no_ext = Path(file).stem
            images[file_no_ext] = {'original': os.path.join(main_dir, file)}
            for model in model_names:
                model_path = os.path.join(main_dir, model, f"prediction_{file}")
                if os.path.exists(model_path):
                    images[file_no_ext][model] = model_path
                else:
                    # Try replacing the file extension with .jpg
                    jpg_model_path = model_path.rsplit('.', 1)[0] + '.jpg'
                    if os.path.exists(jpg_model_path):
                        images[file_no_ext][model] = jpg_model_path
    return images, model_names

def toggle_model_display():
    update_image()

def next_image():
    global current_image
    current_image = (current_image + 1) % len(image_list)
    update_image()

def prev_image():
    global current_image
    current_image = (current_image - 1) % len(image_list)
    update_image()

def update_image():
    file_no_ext = image_list[current_image]
    image_data = images[file_no_ext]
    image_number_label.config(text=f"{current_image + 1} / {len(image_list)}")

    for idx, model in enumerate(['original'] + model_names):
        if model_display_vars[idx].get():
            image_path = image_data.get(model)
            if image_path:
                img = Image.open(image_path)
                img.thumbnail((400, 400), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                label_images[idx].config(image=img)
                label_images[idx].image = img  # Keep a reference
            else:
                label_images[idx].config(image='')
                if model != 'original':
                    print(f"Missing inference for '{file_no_ext}' by model '{model}'")
        else:
            label_images[idx].config(image='')
        label_texts[idx].config(text=model)

default_dir = "/mnt/c/Users/Fistus/Desktop/inference_test_data"

main_dir = input(f"Enter the directory path [{default_dir}]: ")
main_dir = main_dir if main_dir else default_dir

images, model_names = load_images(main_dir)
image_list = list(images.keys())
current_image = 0

root = tk.Tk()
root.title("Model Inference Comparison")

image_number_label = tk.Label(root, text="")
image_number_label.grid(row=0, column=0, columnspan=len(model_names) + 1)

label_images = [tk.Label(root) for _ in range(len(model_names) + 1)]
label_texts = [tk.Label(root, text="") for _ in range(len(model_names) + 1)]
model_display_vars = [tk.BooleanVar(root, True) for _ in range(len(model_names) + 1)]

for idx in range(len(model_names) + 1):
    label_images[idx].grid(row=1, column=idx, padx=10, pady=10)
    label_texts[idx].grid(row=2, column=idx)
    tk.Checkbutton(root, variable=model_display_vars[idx], command=toggle_model_display).grid(row=3, column=idx)

button_prev = tk.Button(root, text="Previous", command=prev_image)
button_prev.grid(row=4, column=0, padx=10, pady=10)

button_next = tk.Button(root, text="Next", command=next_image)
button_next.grid(row=4, column=1, padx=10, pady=10)

update_image()

root.mainloop()
