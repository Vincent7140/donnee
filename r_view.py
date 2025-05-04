import json
import math
import os
from copy import deepcopy

def generate_orbit(center_lon, center_lat, radius_deg, n_views, altitude_offset=0):
    """
    Génère des positions de caméra sur une orbite autour du point (center_lon, center_lat).
    """
    camera_positions = []
    for i in range(n_views):
        angle_deg = i  
        angle_rad = math.radians(angle_deg)
        lon = center_lon + radius_deg * math.cos(angle_rad)
        lat = center_lat + radius_deg * math.sin(angle_rad)
        camera_positions.append({
            "lon": lon,
            "lat": lat,
            "alt": altitude_offset  
        })
    return camera_positions

def main():
    input_json_path = "JAX_214_007_RGB.json"
    output_dir = "orbit_views"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json_path, "r") as f:
        base_data = json.load(f)

    center_lon, center_lat = base_data["geojson"]["center"]
    radius_deg = 0.0003  
    n_views = 90

    camera_positions = generate_orbit(center_lon, center_lat, radius_deg, n_views)

    for i, cam_pos in enumerate(camera_positions):
        new_data = deepcopy(base_data)
        new_data["geojson"]["center"] = [cam_pos["lon"], cam_pos["lat"]]
        new_data["virtual_camera_pose"] = cam_pos
        new_data["simulated"] = True
        output_path = os.path.join(output_dir, f"JAX_214_007_RGB_orbit_{i:03d}.json")
        with open(output_path, "w") as out_f:
            json.dump(new_data, out_f, indent=2)

    print(f"{n_views} fichiers JSON générés dans le dossier '{output_dir}'.")

if __name__ == "__main__":
    main()
