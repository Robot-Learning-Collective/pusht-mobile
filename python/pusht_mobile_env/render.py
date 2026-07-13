"""The canonical PushT frame renderer.

Both the LeRobot dataset export (scripts/export_lerobot.py) and the gymnasium
env observations use this exact code, so training images and eval images are
pixel-identical by construction. Style matches gym-pusht: white background,
goal T in light green, block T in gray, agent in royal blue.
"""

import math

import numpy as np
from PIL import Image, ImageDraw

WORLD = 512
IMG = 96  # lerobot/pusht image size

# T outline in body frame at unit scale (same as T_OUTLINE in pusht-core).
T_OUTLINE = [(-60, -45), (60, -45), (60, -15), (15, -15), (15, 75), (-15, 75), (-15, -15), (-60, -15)]


def t_polygon(x: float, y: float, angle: float, scale: float) -> list[tuple[float, float]]:
    ca, sa = math.cos(angle), math.sin(angle)
    return [(x + (ca * px - sa * py) * scale, y + (sa * px + ca * py) * scale) for px, py in T_OUTLINE]


def render_frame(
    agent: tuple[float, float],
    block: tuple[float, float, float],
    goal: tuple[float, float, float],
    scale: float,
    agent_radius: float,
    size: int = IMG,
) -> np.ndarray:
    """Render one frame to an RGB uint8 array of shape (size, size, 3)."""
    im = Image.new("RGB", (WORLD, WORLD), "white")
    d = ImageDraw.Draw(im)
    d.polygon(t_polygon(goal[0], goal[1], goal[2], scale), fill="lightgreen")
    d.polygon(t_polygon(block[0], block[1], block[2], scale), fill="gray")
    x, y = agent
    r = agent_radius
    d.ellipse([x - r, y - r, x + r, y + r], fill="royalblue")
    return np.asarray(im.resize((size, size), Image.LANCZOS))
