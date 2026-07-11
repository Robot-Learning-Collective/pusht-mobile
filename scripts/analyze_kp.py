"""Determine the canonical body-frame template for the 8 keypoints, in the
exact index order the dataset uses. We use frame 0 of an episode: estimate the
T orientation from the bar's top edge, un-rotate, and print body-frame coords."""
import json
import numpy as np

ep = json.load(open("data/episodes.json"))["episodes"][0]
kp = np.array(ep["keypoints"][0]).reshape(8, 2)

# Try several frames/episodes to confirm the template is constant.
data = json.load(open("data/episodes.json"))["episodes"]

def template_from(kp):
    # bar top edge kp0->kp1 has length 120 in our earlier check
    theta = np.arctan2(kp[1, 1] - kp[0, 1], kp[1, 0] - kp[0, 0])
    c, s = np.cos(-theta), np.sin(-theta)
    R = np.array([[c, -s], [s, c]])
    centroid = kp.mean(axis=0)
    local = (kp - centroid) @ R.T  # un-rotate about centroid
    return local

base = template_from(kp)
print("body-frame template (centroid-relative, bar aligned to x), index order:")
for i, (x, y) in enumerate(base):
    print(f"  kp{i}: ({x:8.3f}, {y:8.3f})")

# verify constant across episodes/frames
maxdev = 0.0
for e in data:
    K = np.array(e["keypoints"])
    for f in range(0, len(K), max(1, len(K)//5)):
        t = template_from(K[f].reshape(8, 2))
        maxdev = max(maxdev, np.abs(t - base).max())
print(f"max deviation of template across sampled frames/eps: {maxdev:.4f}")

# all pairwise distances of the base template, to identify the shape
print("\npairwise distances:")
for i in range(8):
    row = " ".join(f"{np.hypot(*(base[i]-base[j])):6.1f}" for j in range(8))
    print(f"  {i}: {row}")
