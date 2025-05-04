import json
import math
import os
from copy import deepcopy

def generate_inclined_directions(n_views):
    """
    Génère n_views directions de visée avec inclinaison croissante.
    """
    directions = []
    for i in range(n_views):
        theta_deg = i  # angle d'inclinaison
        theta_rad = math.radians(theta_deg)
        dx = math.sin(theta_rad)
        dz = -math.cos(theta_rad)  # vers le bas
        directions.append([dx, 0, dz])  # On regarde dans le plan XZ uniquement
    return directions

def main():
    input_json = "JAX_214_007_RGB.json"
    output_dir = "inclined_views"
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r") as f:
        base_data = json.load(f)

    n_views = 90
    cam_alt = 1000  # position fixe de la caméra en mètres
    center_lon, center_lat = base_data["geojson"]["center"]

    directions = generate_inclined_directions(n_views)

    for i, direction in enumerate(directions):
        new_data = deepcopy(base_data)

        # Position caméra au-dessus du centre
        virtual_pose = {
            "lon": center_lon,
            "lat": center_lat,
            "alt": cam_alt
        }

        new_data["virtual_camera_pose"] = virtual_pose
        new_data["view_direction"] = {
            "vector": direction,
            "theta_deg": i
        }
        new_data["simulated"] = True
        new_data["geojson"]["center"] = [center_lon, center_lat]  # pas de déplacement horizontal

        output_path = os.path.join(output_dir, f"JAX_214_007_RGB_incline_{i:03d}.json")
        with open(output_path, "w") as f_out:
            json.dump(new_data, f_out, indent=2)

    print(f"{n_views} vues inclinées générées dans '{output_dir}'.")

if __name__ == "__main__":
    main()
