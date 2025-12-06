# gui.py
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import time
import os

# Import your modules
from helpers import preprocess_image
from model_loader import predict_single

# ... (keep all your existing color definitions and GUI setup)

colors = {
    "bg": "white smoke",
    "dark": "navy",
    "light": "white",
    "text_muted": "gray50",
    "primary": "deep sky blue",
    "success": "sea green",
    "danger": "red3",
}

root = tk.Tk()
root.title("Brain Tumor Classifier")

icon = tk.PhotoImage(file='./assets/Logo.png')
root.iconphoto(True, icon)

root.geometry("1280x960")
root.configure(bg=colors["bg"])

# Title
title_label = tk.Label(root, text="Brain Tumor Classifier", font=("Arial", 24, "bold"), 
                        bg=colors["bg"], fg=colors["dark"])
title_label.pack(pady=(15, 10))

# Frame for image display
image_frame = tk.Frame(root, bg=colors["bg"])
image_frame.pack(pady=10)

original_frame = tk.LabelFrame(image_frame, text="Original Image", 
                                font=("Arial", 12, "bold"), bg=colors["bg"], 
                                relief="groove", bd=2)
original_frame.grid(row=0, column=0, padx=15, pady=5)

processed_frame = tk.LabelFrame(image_frame, text="Processed Image", 
                                font=("Arial", 12, "bold"), bg=colors["bg"],
                                relief="groove", bd=2)
processed_frame.grid(row=0, column=1, padx=15, pady=5)

# Labels to hold images
original_img_label = tk.Label(original_frame, bg=colors["light"])
original_img_label.pack(padx=10, pady=10)

processed_img_label = tk.Label(processed_frame, bg=colors["light"])
processed_img_label.pack(padx=10, pady=10)

# Set default image size
IMG_WIDTH, IMG_HEIGHT = 400, 400

# Default placeholder
default_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='lightgray')
default_tk = ImageTk.PhotoImage(default_img)
original_img_label.configure(image=default_tk)
original_img_label.image = default_tk
processed_img_label.configure(image=default_tk)
processed_img_label.image = default_tk

# Progress bar
progress_frame = tk.Frame(root, bg=colors["bg"])
progress_frame.pack(pady=15)

progress_label = tk.Label(progress_frame, text="Progress:", font=("Arial", 11), bg=colors["bg"])
progress_label.grid(row=0, column=0, padx=(0, 10))

progress = ttk.Progressbar(progress_frame, orient="horizontal", length=300, mode="determinate")
progress.grid(row=0, column=1)

# Pipeline selection
pipeline_frame = tk.Frame(root, bg=colors["bg"])
pipeline_frame.pack(pady=10)

pipeline_label = tk.Label(pipeline_frame, text="Select Pipeline:", 
                          font=("Arial", 11, "bold"), bg=colors["bg"])
pipeline_label.grid(row=0, column=0, padx=10)

pipeline_var = tk.StringVar(value="cnn")
pipeline_options = ["cnn", "knn", "kmeans", "cnn_knn", "kmeans_knn"]
pipeline_menu = ttk.Combobox(pipeline_frame, textvariable=pipeline_var, 
                             values=pipeline_options, state="readonly", width=15)
pipeline_menu.grid(row=0, column=1, padx=10)

# Results display
results_frame = tk.Frame(root, bg=colors["bg"], relief="ridge", bd=2)
results_frame.pack(pady=20, padx=40, fill="x")

result_title = tk.Label(results_frame, text="Prediction Result", 
                        font=("Arial", 16, "bold"), bg=colors["bg"])
result_title.pack(pady=(10, 5))

result_label = tk.Label(results_frame, text="No prediction yet", 
                        font=("Arial", 22, "bold"), bg=colors["bg"], fg=colors["text_muted"])
result_label.pack(pady=10)

confidence_label = tk.Label(results_frame, text="", font=("Arial", 14), bg=colors["bg"])
confidence_label.pack(pady=5)

selected_image_path = None

# --------------------------
# Load Image
# --------------------------
def load_image():
    global selected_image_path
    
    path = filedialog.askopenfilename(
        title="Select Brain MRI Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.tif *.bmp *.gif")]
    )
    if not path:
        return
    
    selected_image_path = path

    try:
        img = Image.open(path)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        
        original_img_label.configure(image=img_tk)
        original_img_label.image = img_tk
        
        processed_img_label.configure(image=default_tk)
        processed_img_label.image = default_tk
        result_label.config(text="Image Loaded - Ready for Prediction", 
                          fg=colors["primary"])
        confidence_label.config(text="")
        
        predict_btn.config(state="normal")
        
        filename = os.path.basename(path)
        status_label.config(text=f"Loaded: {filename}")
        
    except Exception as e:
        result_label.config(text=f"Error loading image: {str(e)}", fg="red")

# --------------------------
# Predict
# --------------------------
def predict_image():
    if not selected_image_path:
        result_label.config(text="Please load an image first!", fg="orange")
        return
    
    try:
        progress["value"] = 0
        progress_label.config(text="Processing...")
        root.update_idletasks()
        
        # Preprocess
        progress["value"] = 30
        root.update_idletasks()
        
        processed = preprocess_image(selected_image_path)
        
        if processed is None:
            raise ValueError("Preprocessing failed")
        
        # Show processed image
        progress["value"] = 50
        root.update_idletasks()
        
        if hasattr(processed, 'shape'):
            if len(processed.shape) == 4:
                processed_show = processed[0].squeeze()
            else:
                processed_show = processed.squeeze()
            
            if processed_show.max() <= 1.0:
                processed_show = (processed_show * 255).astype(np.uint8)
            else:
                processed_show = processed_show.astype(np.uint8)
            
            if len(processed_show.shape) == 2:
                processed_img = Image.fromarray(processed_show, mode='L')
                processed_img = processed_img.convert('RGB')
            else:
                processed_img = Image.fromarray(processed_show)
            
            processed_img = processed_img.resize((IMG_WIDTH, IMG_HEIGHT), 
                                                 Image.Resampling.LANCZOS)
            processed_tk = ImageTk.PhotoImage(processed_img)
            
            processed_img_label.configure(image=processed_tk)
            processed_img_label.image = processed_tk
        
        # Prediction with selected pipeline
        progress["value"] = 70
        progress_label.config(text="Analyzing...")
        root.update_idletasks()
        
        selected_pipeline = pipeline_var.get()
        pred, prob = predict_single(processed, pipeline=selected_pipeline)
        
        progress["value"] = 100
        progress_label.config(text="Complete!")
        root.update_idletasks()
        
        # Display results
        if pred == 1:
            result_label.config(text="⚠️ Tumor Detected", fg=colors["danger"])
        else:
            result_label.config(text="✅ Healthy", fg=colors["success"])
        
        confidence_label.config(text=f"Confidence: {prob*100:.2f}% | Pipeline: {selected_pipeline.upper()}")
        
        root.after(1000, lambda: progress.configure(value=0))
        root.after(1000, lambda: progress_label.config(text="Progress:"))
        
        predict_btn.config(state="disabled")
        root.after(1200, lambda: predict_btn.config(state="normal"))
        
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error: {str(e)}")
        result_label.config(text=f"Error: {str(e)}", fg="red")
        progress["value"] = 0
        progress_label.config(text="Progress:")

# --------------------------
# Clear all
# --------------------------
def clear_all():
    global selected_image_path
    selected_image_path = None
    
    original_img_label.configure(image=default_tk)
    original_img_label.image = default_tk
    processed_img_label.configure(image=default_tk)
    processed_img_label.image = default_tk
    
    result_label.config(text="No prediction yet", fg=colors["text_muted"])
    confidence_label.config(text="")
    progress["value"] = 0
    progress_label.config(text="Progress:")
    status_label.config(text="Ready")
    predict_btn.config(state="normal")

# Buttons
btn_frame = tk.Frame(root, bg=colors["bg"])
btn_frame.pack(pady=15)

load_btn = tk.Button(btn_frame, text="📁 Load Image", width=15, 
                      command=load_image, bg=colors["primary"], fg=colors["light"],
                      font=("Arial", 11, "bold"), relief="raised")
load_btn.grid(row=0, column=0, padx=10, pady=5)

predict_btn = tk.Button(btn_frame, text="🔍 Predict", width=15, 
                        command=predict_image, bg=colors["success"], fg=colors["light"],
                        font=("Arial", 11, "bold"), relief="raised", state="normal")
predict_btn.grid(row=0, column=1, padx=10, pady=5)

clear_btn = tk.Button(btn_frame, text="🗑️ Clear All", width=15, 
                      command=clear_all, bg=colors["danger"], fg=colors["light"],
                      font=("Arial", 11, "bold"), relief="raised")
clear_btn.grid(row=0, column=2, padx=10, pady=5)

# Status bar
status_frame = tk.Frame(root, bg=colors["dark"], height=25)
status_frame.pack(side="bottom", fill="x")

status_label = tk.Label(status_frame, text="Ready", bg=colors["dark"], 
                        fg=colors["light"], anchor="w", padx=10)
status_label.pack(fill="both")

# Instructions
instructions = """
Instructions:
1. Select a pipeline (CNN recommended for best accuracy)
2. Click 'Load Image' to select a brain MRI scan
3. Click 'Predict' to analyze for tumors
4. View results and confidence score
"""
instructions_label = tk.Label(root, text=instructions, bg=colors["bg"], 
                              font=("Arial", 9), justify="left", fg=colors["text_muted"])
instructions_label.pack(pady=10)

root.resizable(False, False)
root.mainloop()