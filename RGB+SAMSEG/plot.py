import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Define the parameters for MSCAN
stages = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
embed_dims = [64, 128, 320, 512]
depths = [3, 5, 27, 3]
resolutions = ["H/4 x W/4", "H/8 x W/8", "H/16 x W/16", "H/32 x W/32"]

# Initialize the figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 8)
ax.set_ylim(0, 5)
ax.axis("off")

# Draw each stage as a box
for i, (stage, embed_dim, depth, resolution) in enumerate(zip(stages, embed_dims, depths, resolutions)):
    y = 4 - i  # Vertical position
    x = 2      # Horizontal starting position
    # Draw the stage box
    box = FancyBboxPatch((x, y - 0.5), 4, 0.8, boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue")
    ax.add_patch(box)
    # Add text inside the box
    ax.text(x + 2, y - 0.1, f"{stage}\nDepth: {depth}\nChannels: {embed_dim}\nResolution: {resolution}",
            ha="center", va="center", fontsize=10)

# Add input and output annotations
ax.text(0.5, 4.5, "Input\nImage", ha="center", va="center", fontsize=12, color="green")
ax.text(7.5, 0.5, "Output\nFeatures", ha="center", va="center", fontsize=12, color="red")

# Draw arrows connecting stages
for i in range(4):
    start_x = 6
    end_x = 7
    y_pos = 4 - i
    # Stage-to-stage arrow
    ax.arrow(start_x - 1.5, y_pos - 0.1, 1.5, 0, head_width=0.2, head_length=0.3, fc="black", ec="black")

# Input arrow
ax.arrow(0.8, 4.4, 1.2, 0, head_width=0.2, head_length=0.3, fc="black", ec="black")

# Output arrow
ax.arrow(6.2, 0.5, 1.2, 0, head_width=0.2, head_length=0.3, fc="black", ec="black")

# Save the plot as a file
plt.tight_layout()
output_filename = "mscan_flowchart.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Flowchart saved as {output_filename}")

# Show the plot
plt.show()
