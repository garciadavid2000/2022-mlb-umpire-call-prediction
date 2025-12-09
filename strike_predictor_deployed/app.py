import gradio as gr
import torch
from torch import nn
import lightning as L
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler

class BaseModel(L.LightningModule):
    def __init__(self, num_classes, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()
    
    def forward(self, x):
        return self.model(x).squeeze(1)

class MLP(BaseModel):
    def __init__(self, num_classes, hidden=256, learning_rate=0.001, dropout_rate=0.2):
        self.hidden = hidden
        self.dropout_rate = dropout_rate
        super().__init__(num_classes, learning_rate)

    def build_model(self):
        return nn.Sequential(
            nn.Linear(11, self.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, 1)
        )

try:
    scaler = joblib.load('scaler/scaler.pkl')
    print("Scaler loaded successfully.")
except:
    print("Warning: 'scaler.pkl' not found.")
    scaler = None

try:
    model = MLP.load_from_checkpoint(
            "checkpoint/epoch=9-step=7970.ckpt",
            strict=False,
            num_classes=1,
            hidden=256
        )

    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Warning: Model not loaded. Error: {e}")

def generate_pitch_plot(plate_x, plate_z, sz_top, sz_bot, is_strike, probability, pfx_x, pfx_z, release_speed, balls, strikes):
    """
    Creates a visual representation of the pitch, its movement, and game stats.
    """
    fig, ax = plt.subplots(figsize=(5, 6)) # Increased height slightly for text at bottom
    
    plate_width_ft = 17 / 12
    half_width = plate_width_ft / 2
    
    strike_zone = patches.Rectangle(
            (-half_width, sz_bot), 
            plate_width_ft, 
            sz_top - sz_bot, 
            linewidth=2, 
            edgecolor='black', 
            facecolor='none', 
            linestyle='--'
        )
    ax.add_patch(strike_zone)
    
    # This assumes pfx_x and pfx_z are in the same units (ft) as plate_x/z
    start_x = plate_x - pfx_x
    start_z = plate_z + pfx_z

    # 3. Plot the Elements
    # The actual final pitch location
    pitch_color = 'red' if is_strike else 'blue'
    ax.scatter(plate_x, plate_z, s=150, c=pitch_color, edgecolors='black', label='Final Position', zorder=3)
    
    # The "No Movement" location (Ghost dot)
    ax.scatter(start_x, start_z, s=100, c='gray', alpha=0.5, edgecolors='black', label='Initial Position', zorder=2)
    
    # Draw arrow from No-Move -> Final
    ax.annotate("", 
                xy=(plate_x, plate_z), 
                xytext=(start_x, start_z),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5, ls='-'))

    # 4. Axes Setup
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-1, 6.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Horizontal Location (ft)')
    ax.set_ylabel('Vertical Location (ft)')
    ax.grid(True, alpha=0.3)
    
    # 5. Legend
    # We force the legend to show our specific labels
    ax.legend(loc='upper right', fontsize='small')
    
    # 6. Title and Info Text
    title_text = f"Prediction: {'STRIKE' if is_strike else 'BALL'} {probability:.1%}"
    ax.set_title(title_text, fontsize=14, fontweight='bold', color=pitch_color)

    # Add text info underneath the graph
    info_text = (f"Speed: {release_speed:.1f} mph\n"
                 f"Count: {int(balls)}-{int(strikes)}\n"
                 f"Break: {pfx_x:.1f}' Horz, {pfx_z:.1f}' Vert")
    
    # Place text box in figure coords (0,0 is bottom left, 1,1 is top right)
    plt.figtext(0.5, 0.02, info_text, ha='center', fontsize=10, 
                bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make room for text at bottom
    return fig

def predict_strike(plate_x, plate_z, sz_top, sz_bot, balls, strikes, release_speed, pfx_x, pfx_z, p_throws, stand):

    data = {
        "plate_x": [plate_x], "plate_z": [plate_z],
        "sz_top": [sz_top], "sz_bot": [sz_bot],
        "balls": [balls], "strikes": [strikes],
        "release_speed": [release_speed],
        "pfx_x": [pfx_x], "pfx_z": [pfx_z],
        "p_throws": [p_throws], "stand": [stand]
    }
    df = pd.DataFrame(data)

    df["p_throws"] = df["p_throws"].map({"L": 0, "R": 1})
    df["stand"] = df["stand"].map({"L": 0, "R": 1})

    numeric_features = ["plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes", "release_speed", "pfx_x", "pfx_z"]

    if scaler:
        df[numeric_features] = scaler.transform(df[numeric_features])
    
    input_tensor = torch.tensor(df.values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
    
    is_strike = probability > 0.5
    classification = "STRIKE" if is_strike else "BALL"
    text_output = f"{classification} ({probability:.1%})"
    
    plot_output = generate_pitch_plot(
        plate_x, plate_z, sz_top, sz_bot, is_strike, probability,
        pfx_x, pfx_z, release_speed, balls, strikes
    )
    
    return text_output, plot_output

inputs = [
    gr.Slider(-3.5, 3.5, step=0.1, label="Plate X (Horizontal Pos)", value=0.0),
    gr.Slider(0.0, 6.0, step=0.1, label="Plate Z (Vertical Pos)", value=2.5),
    gr.Slider(2.5, 4.5, step=0.1, label="Strike Zone Top", value=3.4),
    gr.Slider(0.5, 2.5, step=0.1, label="Strike Zone Bottom", value=1.5),
    gr.Slider(0, 3, step=1, label="Balls"),
    gr.Slider(0, 2, step=1, label="Strikes"),
    gr.Slider(40, 105, step=1, label="Release Speed (mph)", value=90.0),
    gr.Slider(-2.5, 2.5, step=0.1, label="PFX X (Horizontal Movement)", value=0.0),
    gr.Slider(-2.5, 2.5, step=0.1, label="PFX Z (Vertical Movement)", value=0.0),
    gr.Radio(["L", "R"], label="Pitcher Handedness", value="R"),
    gr.Radio(["L", "R"], label="Batter Stance", value="R"),
]

demo = gr.Interface(
    fn=predict_strike,
    inputs=inputs,
    outputs=["text", "plot"],
    title="Baseball Strike Predictor",
    description="Predicts if a pitch is a ball or a strike. The plot visualizes the pitch location relative to the batter's strike zone."
)

if __name__ == "__main__":
    demo.launch()