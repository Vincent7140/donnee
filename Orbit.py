def generate_virtual_rays(cam_pos, look_at, H, W, fov, near, far):
    """
    Génère des rayons depuis une caméra virtuelle orientée vers 'look_at'.
    Args:
        cam_pos: position ECEF de la caméra
        look_at: point ECEF à viser
        H, W: taille de l'image virtuelle
        fov: champ de vision (en degrés)
        near, far: bornes en mètres
    Return:
        rays: (H*W, 8) tensor [origin (3), direction (3), near, far]
    """
    import torch
    import numpy as np

    cam_dir = look_at - cam_pos
    cam_dir = cam_dir / np.linalg.norm(cam_dir)

    # définir un repère local
    up = np.array([0, 0, 1])
    right = np.cross(cam_dir, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, cam_dir)
    up /= np.linalg.norm(up)

    # grille de pixels
    i, j = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    i = i.reshape(-1)
    j = j.reshape(-1)

    # coordonnées dans le plan image
    fov_rad = np.radians(fov)
    aspect = W / H
    x = (2 * (i + 0.5) / W - 1) * np.tan(fov_rad / 2) * aspect
    y = (1 - 2 * (j + 0.5) / H) * np.tan(fov_rad / 2)

    # directions dans l’espace
    dirs = x[:, None] * right + y[:, None] * up + cam_dir[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    rays_o = np.repeat(cam_pos[None, :], H * W, axis=0)
    nears = np.full((H * W, 1), near)
    fars = np.full((H * W, 1), far)

    rays = np.hstack([rays_o, dirs, nears, fars])
    return torch.from_numpy(rays).float()


# définir une orbite synthétique
center_ecef = dataset.center.numpy()
radius = 0.2  # en unités normalisées ou absolues selon ton besoin
height = 0.1
n_views = 8
fov = 40  # deg

virtual_rays_list = []
for i in range(n_views):
    theta = 2 * np.pi * i / n_views
    x = center_ecef[0] + radius * np.cos(theta)
    y = center_ecef[1] + radius * np.sin(theta)
    z = center_ecef[2] + height
    cam_pos = np.array([x, y, z])
    rays = generate_virtual_rays(cam_pos, center_ecef, H=128, W=128, fov=fov, near=0, far=1)
    virtual_rays_list.append(rays.cuda())



for v_idx, rays_virtual in enumerate(virtual_rays_list):
    results_virtual = batched_inference(models, rays_virtual, ts=None, args=args)
    # sauver les images virtuelles
    img_virtual = results_virtual['rgb_fine' if 'rgb_fine' in results_virtual else 'rgb_coarse']
    img_virtual = img_virtual.view(128, 128, 3).permute(2, 0, 1).cpu()
    path = f"{out_dir}/virtual_rgb/view{v_idx:02d}_epoch{epoch_number}.tif"
    train_utils.save_output_image(img_virtual, path, src_path=None)
