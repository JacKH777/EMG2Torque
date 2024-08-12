import numpy as np
import math

def decode2torque(angle, force):
    angle_2_rad = math.radians(angle)  # 将角度转换为弧度
    c = math.sqrt(a**2 + b**2 - 2 * a * b * math.cos(angle_2_rad))
    return c