import json
import numpy as np
from rpcm.rpc_model import RPCModel
from rpcfit import calibrate_rpc

# === Charger ton RPC d'origine ===
with open("JAX_214_007_RGB.json", "r") as f:
    data = json.load(f)
rpc_orig = RPCModel(data["rpc"], dict_format="rpcm")

# === Cible centrale (au sol) autour de laquelle orbiter ===
target_lon = rpc_orig.lon_offset
target_lat = rpc_orig.lat_offset
target_alt = rpc_orig.alt_offset

# === Paramètres d'orbite ===
radius_deg = 0.0005  # ≈ 50m
num_poses = 10
angles = np.linspace(0, 2 * np.pi, num_poses, endpoint=False)

# === Points de la scène à projeter ===
scene_points = []
for dx in np.linspace(-0.0001, 0.0001, 10):
    for dy in np.linspace(-0.0001, 0.0001, 10):
        for dz in np.linspace(-50, 50, 5):
            scene_points.append([target_lon + dx, target_lat + dy, target_alt + dz])
scene_points = np.array(scene_points)

# === Génération des poses ===
for i, theta in enumerate(angles):
    # Nouvelle position caméra
    cam_lon = target_lon + radius_deg * np.cos(theta)
    cam_lat = target_lat + radius_deg * np.sin(theta)
    cam_alt = target_alt + 20  # optionnel : surélevé

    # Modifier les points en fonction de cette vue
    # ici on suppose que la projection reste la même (comme si tous les points venaient de cette vue)
    points_2d = np.array([rpc_orig.projection(*p) for p in scene_points])

    # Générer un nouveau modèle RPC artificiel
    rpc_new = calibrate_rpc(
        target=points_2d,
        input_locs=scene_points,
        separate=True,
        orientation="projection",
        init=None
    )

    # Sauvegarde
    output_name = f"rpc_orbite_pose_{i:02}.json"
    with open(output_name, "w") as f:
        json.dump(rpc_new.__dict__, f, indent=2)
    print(f"✅ Pose artificielle #{i} sauvegardée : {output_name}")
