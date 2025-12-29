import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path

# Model Definition (MUST match training)
# -----------------------------
class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = block(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = block(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

# Configuration
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "unet_deepglobe.pth"

COLORS = np.array([
    [0, 0, 0],       # Unknown
    [0, 255, 255],   # Urban
    [255, 255, 0],   # Agriculture
    [255, 0, 255],   # Rangeland
    [0, 255, 0],     # Forest
    [0, 0, 255],     # Water
    [255, 255, 255]  # Barren
], dtype=np.uint8)

# Load Model
# -----------------------------
model = UNet(num_classes=7).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Helper Functions
# -----------------------------
def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
    return image, tensor.unsqueeze(0)

def colorize_mask(mask):
    return COLORS[mask]

# UI Functions
# -----------------------------
def upload_image():
    global raw_image, raw_image_pil
    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not path:
        return

    raw_image, tensor = preprocess_image(path)
    raw_image_pil = Image.fromarray(raw_image)
    raw_image_pil = raw_image_pil.resize((350, 350))

    img_tk = ImageTk.PhotoImage(raw_image_pil)
    raw_canvas.image = img_tk
    raw_canvas.create_image(175, 175, image=img_tk)

    process_btn.config(state="normal")

def process_image():
    global segmented_pil

    _, tensor = preprocess_image("temp.jpg") if False else (None, None)
    image = np.array(raw_image)
    tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    colored = colorize_mask(pred)
    segmented_pil = Image.fromarray(colored).resize((350, 350))

    img_tk = ImageTk.PhotoImage(segmented_pil)
    seg_canvas.image = img_tk
    seg_canvas.create_image(175, 175, image=img_tk)

    save_btn.config(state="normal")

def save_result():
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        segmented_pil.save(path)
        messagebox.showinfo("Saved", "Segmented image saved successfully.")

# UI Layout
# -----------------------------
root = tk.Tk()
root.title("Land Cover Segmentation System")
root.configure(bg="#f2f2f2")
root.geometry("900x750")

title = tk.Label(
    root, text="Land Cover Segmentation",
    font=("Arial", 18, "bold"),
    bg="#f2f2f2"
)
title.pack(pady=10)

btn_frame = tk.Frame(root, bg="#f2f2f2")
btn_frame.pack(pady=5)

upload_btn = tk.Button(btn_frame, text="Upload", width=12, command=upload_image)
upload_btn.grid(row=0, column=0, padx=10)

process_btn = tk.Button(btn_frame, text="Process", width=12, state="disabled", command=process_image)
process_btn.grid(row=0, column=1, padx=10)

save_btn = tk.Button(btn_frame, text="Save", width=12, state="disabled", command=save_result)
save_btn.grid(row=0, column=2, padx=10)

canvas_frame = tk.Frame(root, bg="#f2f2f2")
canvas_frame.pack(pady=20)

raw_canvas = tk.Canvas(canvas_frame, width=350, height=350, bg="#e6e6e6")
raw_canvas.grid(row=0, column=0, padx=20)

seg_canvas = tk.Canvas(canvas_frame, width=350, height=350, bg="#e6e6e6")
seg_canvas.grid(row=0, column=1, padx=20)

# Legend
# -----------------------------
legend_frame = tk.Frame(root, bg="#f2f2f2")
legend_frame.pack(pady=15)

legend_title = tk.Label(
    legend_frame, text="Land-Cover Class Legend",
    font=("Arial", 12, "bold"), bg="#f2f2f2"
)
legend_title.pack()

legend_items = [
    ("Urban", "#00ffff"),
    ("Agriculture", "#ffff00"),
    ("Rangeland", "#ff00ff"),
    ("Forest", "#00ff00"),
    ("Water", "#0000ff"),
    ("Barren", "#ffffff"),
    ("Unknown", "#000000"),
]

legend_grid = tk.Frame(legend_frame, bg="#f2f2f2")
legend_grid.pack(pady=5)

for i, (name, color) in enumerate(legend_items):
    box = tk.Label(legend_grid, bg=color, width=2, height=1)
    box.grid(row=i//4, column=(i%4)*2, padx=5, pady=5)
    label = tk.Label(legend_grid, text=name, bg="#f2f2f2")
    label.grid(row=i//4, column=(i%4)*2+1, padx=5, pady=5)

root.mainloop()
