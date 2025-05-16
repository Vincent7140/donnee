
from rpcfit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# Charger le fichier RPC existant
with open("JAX_214_007_RGB.json") as f:
    import json
    data = json.load(f)

rpc = RPCModel(data["rpc"], dict_format="rpcm")

# Générer des points 3D autour du centre
N = 500
center_lon, center_lat, center_alt = rpc.lon_offset, rpc.lat_offset, rpc.alt_offset
lon = center_lon + np.random.uniform(-rpc.lon_scale, rpc.lon_scale, N)
lat = center_lat + np.random.uniform(-rpc.lat_scale, rpc.lat_scale, N)
alt = center_alt + np.random.uniform(-rpc.alt_scale/10, rpc.alt_scale/10, N)
points_3d = np.stack([lon, lat, alt], axis=1)

# Projeter avec le modèle original
points_2d = np.array([rpc.projection(*p) for p in points_3d])
points_2d = np.flip(points_2d, axis=1)  # convert (col, row) to (row, col)

# === 4. Générer un nouveau modèle RPC ===
rpc_new = calibrate_rpc(
    target=points_2d,        # (col, row)
    input_locs=points_3d,    # (lon, lat, alt)
    separate=True,
    orientation="projection",
    init=None
)

# === 5. Sauvegarde manuelle (convert to dict and save) ===
with open("rpc_synthetique.json", "w") as f:
    json.dump(rpc_new.__dict__, f, indent=2)

print("✅ Nouveau modèle RPC calibré et sauvegardé.")








import numpy as np
import rpcm
from rpcfit import fit_rpc
from rpcfit.utils import save_rpc

# === 1. Charger la pose RPC initiale ===
from rpcm.rpc_model import RPCModel

rpc_path = "rpc_original.json"  # ton fichier RPC initial
rpc = RPCModel(rpc_path)

# === 2. Générer des points 3D autour du centre de projection ===
center_lon = rpc.lon_offset
center_lat = rpc.lat_offset
center_alt = rpc.alt_offset

# Génère une grille de points (lon, lat, alt) autour du centre
N = 500  # nombre de points
lon = center_lon + np.random.uniform(-rpc.lon_scale, rpc.lon_scale, N)
lat = center_lat + np.random.uniform(-rpc.lat_scale, rpc.lat_scale, N)
alt = center_alt + np.random.uniform(-rpc.alt_scale/10, rpc.alt_scale/10, N)

points_3d = np.stack([lon, lat, alt], axis=1)

# === 3. Projeter ces points en coordonnées image ===
points_2d = np.array([rpc.project(p) for p in points_3d])  # (row, col)

# === 4. Générer un nouveau modèle RPC à partir des correspondances ===
rpc_new = fit_rpc(
    points_3d,
    points_2d[:, 1],  # cols
    points_2d[:, 0],  # rows
    normalize=True
)

# === 5. Sauvegarder la nouvelle pose RPC ===
save_rpc(rpc_new, "rpc_synthetique.json")
print("✅ Nouvelle pose RPC générée et sauvegardée.")




import numpy as np
from rpcfit import fit_rpc, save_rpc
from rpcm.rpc_model import RPCModel

# Pose initiale
rpc = RPCModel("rpc_initial.json")

# Cible au sol (centre de rotation)
target_lon, target_lat = rpc.lon_offset, rpc.lat_offset
alt = rpc.alt_offset

# Rayon de l'orbite (en degrés, à ajuster)
radius_deg = 0.0005  # ≈ 50 m

# Nombre de poses à générer
N = 20
angles = np.linspace(0, 2*np.pi, N, endpoint=False)

for i, theta in enumerate(angles):
    # Nouvelle position caméra sur orbite
    cam_lon = target_lon + radius_deg * np.cos(theta)
    cam_lat = target_lat + radius_deg * np.sin(theta)
    cam_alt = alt

    # Générer des points 3D autour de la cible
    points_3d = []
    for dx in np.linspace(-0.0001, 0.0001, 10):
        for dy in np.linspace(-0.0001, 0.0001, 10):
            for dz in np.linspace(-50, 50, 5):
                points_3d.append([target_lon + dx, target_lat + dy, alt + dz])
    points_3d = np.array(points_3d)

    # Projeter ces points avec la pose initiale (ou une simplifiée)
    rows_cols = np.array([rpc.project(p) for p in points_3d])

    # Ajuster un nouveau RPC
    rpc_new = fit_rpc(points_3d, rows_cols[:, 1], rows_cols[:, 0], normalize=True)

    # Sauvegarder
    save_rpc(rpc_new, f"rpc_orbite_{i:03}.json")



points_2d = np.array([rpc.projection(p[0], p[1], p[2]) for p in points_3d])  # (col, row)
cols_rows = np.array([rpc.projection(*p) for p in points_3d])
rows_cols = np.flip(cols_rows, axis=1)  # passe de (col, row) à (row, col)
