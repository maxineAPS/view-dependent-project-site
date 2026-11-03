#Gradient descent loop to iteratively optimize heights and colors of heightfield to match desired view-dependent appearance.
#Author: Maxine Perroni-Scharf
#Date: June 2023


import os
import json
import numpy as np
import torch
import copy
import torch
import rendering
import heightfield
import image
import opt
import simannealing
from datetime import datetime


'''
Save the (repeated) heights and colors of the heightfield as .pt tensors and as .npy arrays (the .npy files
can be read by the Blender script without installing PyTorch into Blender).
:param H: heightfield heights tensor (one repeating unit)
:param COL: heightfield colors tensor
:param num_repeats: number of times to grid repeat surface heights
:param output_dir: output folder
:param date_time: timestamp used as file name prefix
:param cam_directions, image_paths, camera_z: cameras, desired images and pixel-plane height, saved so that
    reproject.py can project the images onto the surface with the same camera model
:return: None
'''
def save_surface(H, COL, num_repeats, output_dir, date_time, cam_directions=None, image_paths=None, camera_z=1):
    repeated_H = H.detach().repeat(num_repeats, num_repeats)
    COL = COL.detach()
    prefix = os.path.join(output_dir, date_time)
    torch.save(repeated_H, prefix + '_heights.pt')
    torch.save(COL, prefix + '_colors.pt')
    np.save(prefix + '_heights.npy', repeated_H.numpy())
    np.save(prefix + '_colors.npy', COL.numpy())
    if cam_directions is not None:
        with open(prefix + '_cameras.json', 'w') as f:
            json.dump({'camera_directions': np.asarray(cam_directions, dtype=float).tolist(), 'image_paths': list(image_paths), 'camera_z': float(camera_z)}, f, indent=1)
    print("--------------------------------------------------------- \n Saving surface heights and colors to " + prefix + "_heights/_colors (.pt and .npy)...\n---------------------------------------------------------")


'''
Gradient descent loop to iteratively optimize heights and colors of heightfield to match desired view-dependent appearance.
:param num_epochs: number of iterations
:param cam_directions: list of viewing directions, where each direction is a list of 3 values specifying global x, y and z direction components
:param image_paths: list of image paths specifying desired appearance
:param use_barrier_loss: boolean specifying whether to enforce a min/max height on the heightfield
:param use_neighbor_loss: boolean specifying whether to regularize the surface from neighbor to neighbor heights
:param sim_annealing: boolean specifying whether to perform steps of simulated annealing before beginning gradient descent
:param optim_type: optimizer to use ()"Adam" for Adam optimizer, "SGD" for stochastic gradient descent)
:param heaviside_fn: parameter to specify specific heavisde smooth approximation to use for rendering (choices: erfc, tan, log, circ, circdist, see happrox.py)
:param num_repeats: number of times to grid repeat surface heights
:param num_rays: resolution of images (there will be num_raysxnum_rays camera rays shot at the surface from each camera). This value doubles as heightfield subdivides.
:param output_dir: folder where the optimized heights and colors are saved (created if needed)
:param neighbor_weight: weight of the neighbor loss (v2.0: every bar against its 4 neighbors, wrapping around the repeating unit, averaged over bars)
:param max_height: maximum bar height allowed by the barrier loss (v1.0: 1.5; lower values give shallower surfaces with fewer thin spikes)
:param paper_settings: restore the behavior of the original release (v1.0):
    camera rays start on the fixed plane z = 1, the neighbor loss is opt.neighbor_fn (interior bars, summed)
    and the maximum height is 1.5. By default (v2.0) camera rays start on a plane at the current mean surface
    height, re-centered every 10 steps so that each camera's image stays aligned with the surface it sees,
    and the neighbor loss is opt.neighbor_fn_wrap.
:return: final heightfield colors, final heightfield heights, rendered heightfield appearance from each camera, desired heightfield appearance from each camera
'''
def grad_descent(num_epochs, H, cam_directions, image_paths, use_barrier_loss = True, use_neighbor_loss = False, sim_annealing = True, optim_type = "Adam", heaviside_fn = None, heaviside_k = 0.1, num_repeats = 1, num_rays = 32, output_dir = "models", paper_settings = False, max_height = 1.0, neighbor_weight = 1.0):
  
    COL = heightfield.get_white_COL(H.shape[0]*num_repeats)
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    num_cams = len(cam_directions)
    num_strips = H.shape[0]*num_repeats
    ray_width = 1/num_rays
  
    original_ray_origins = [[], [], [], []]
    ray_origins = copy.deepcopy(original_ray_origins)
    coords = []
  
    schedule = torch.zeros(50) #update colors for 10 steps, then heights for 20 steps
    schedule[10:20] = 1
    schedule[40:50] = 1

    grad_types = schedule.repeat(20)

    if paper_settings:
        max_height = 1.5

    #height of the plane the camera rays start from (pixel centers); see paper_settings
    camera_z = 1 if paper_settings else float(torch.as_tensor(H, dtype=torch.float).double().mean())

    for i in range(num_rays):
        for k in range(num_rays):
            coord = np.array([i*ray_width + ray_width/2, k*ray_width + ray_width/2, camera_z])
            coords.append(coord)

            for q in range(num_cams):   
                curr_origin = coord - 20*cam_directions[q]
                curr_origin[2] = curr_origin[2].item()
                ray_origins[q].append(curr_origin)

    H = torch.as_tensor(H, dtype=torch.float).detach().clone().requires_grad_(True)
    COL = torch.as_tensor(COL, dtype=torch.float).detach().clone().requires_grad_(True)

    ray_packs = []

    for t in range(num_cams):
        ray_packs.append(rendering.process_rays(ray_origins[t], cam_directions[t], num_strips))

    desired_cols = []
    for i in range(num_cams):
        desired_cols.append(image.get_tensor_from_image(image_paths[i], (64, 64), (num_rays, num_rays)))
  
    if sim_annealing:
        #keep the perturbed heights strictly inside the barrier bounds (a height of exactly 0.1 makes the barrier loss infinite)
        sa_bounds = dict(min_height = 0.1 + 1e-4, max_height = max_height - 1e-4) if use_barrier_loss else {}
        H, COL = simannealing.sim_annealing(H, COL, ray_packs, num_cams, num_repeats, heaviside_fn, heaviside_k, desired_cols, **sa_bounds)

    print("--------------------------------------------------------- \n Beginning grad descent...\n---------------------------------------------------------")

    if optim_type == "Adam":
        optimizer = torch.optim.AdamW([H, COL],lr=0.1)
    
    if optim_type == "SGD":
        optimizer = torch.optim.SGD([H, COL],lr=0.1, momentum=0.9)

    for j in range(1, num_epochs):
        
        if j%50 == 0:
      
            save_surface(H, COL, num_repeats, output_dir, date_time, cam_directions, image_paths, camera_z)
        
        change = False
    
        if j%100 == 0:
            H = heightfield.double_resolution(H)
            COL = heightfield.double_resolution(COL)
            H = torch.as_tensor(H, dtype=torch.float).detach().clone()
            COL = torch.as_tensor(COL, dtype=torch.float).detach().clone()
            num_strips = num_strips*2
            desired_cols = []
            
            for q in range(num_cams):
                desired_cols.append(image.get_tensor_from_image(image_paths[q], (64, 64), (num_rays, num_rays)))
            change = True

        if j%100 == 95:
            num_rays = num_rays*2
            ray_width = ray_width/2
            change = True
    
        if j%100 == 95 or j%100 == 0:
            coords = []
            change = True
      
            ray_origins = copy.deepcopy(original_ray_origins) 
            for i in range(num_rays):
                for k in range(num_rays):
                    coord = np.array([i*ray_width + ray_width/2, k*ray_width + ray_width/2, camera_z])
                    coords.append(coord)
                    
                    for q in range(num_cams):    
                        curr_origin = coord - 20*cam_directions[q] 
                        curr_origin[2] = curr_origin[2].item()
                        ray_origins[q].append(curr_origin)

            ray_packs = []
            for t in range(num_cams):
                ray_packs.append(rendering.process_rays(ray_origins[t], cam_directions[t], num_strips))
        
            H.requires_grad = True
            COL.requires_grad = True
      
            desired_cols = []
            for i in range(num_cams):
                desired_cols.append(image.get_tensor_from_image(image_paths[i], (64, 64), (num_rays, num_rays)))

        #keep the camera rays starting at the current mean surface height (every 10 steps)
        if not paper_settings and j%10 == 0:
            new_camera_z = float(H.detach().double().mean())
            if abs(new_camera_z - camera_z) > 1e-4:
                camera_z = new_camera_z
                ray_origins = copy.deepcopy(original_ray_origins)
                for i in range(num_rays):
                    for k in range(num_rays):
                        coord = np.array([i*ray_width + ray_width/2, k*ray_width + ray_width/2, camera_z])
                        for q in range(num_cams):
                            curr_origin = coord - 20*cam_directions[q]
                            ray_origins[q].append(curr_origin)
                ray_packs = []
                for t in range(num_cams):
                    ray_packs.append(rendering.process_rays(ray_origins[t], cam_directions[t], num_strips))
                change = True

        if int(grad_types[j].item()) == 0:
      
            H.requires_grad = False
            COL.requires_grad = True
            lr = 0.05
      
        elif int(grad_types[j].item()) == 1:
            H.requires_grad = True
            COL.requires_grad = False 
            COL = COL.detach().numpy()
            COL = torch.tensor(COL, dtype=torch.float, requires_grad = False)
            if j > 49:
                lr = 0.03
      
        elif int(grad_types[j].item()) == 2:
            H.requires_grad = True
            COL.requires_grad = True
            H = torch.tensor(H, dtype=torch.float, requires_grad = True)
            COL = torch.tensor(COL, dtype=torch.float, requires_grad = True)

        if j > 0:
            if int(grad_types[j].item()) != int(grad_types[j-1].item()):
                change = True
        
        if change:
            if optim_type == "Adam":
                optimizer = torch.optim.AdamW([H, COL],lr=lr)
            if optim_type == "SGD":
                optimizer = torch.optim.SGD([H, COL],lr=lr, momentum=0.9)
    
        result_cols = rendering.render_camera(H, COL, ray_packs, num_cams, num_repeats=num_repeats, heaviside_fn = heaviside_fn, heaviside_k = heaviside_k)
        mse_loss = 0
        for q in range(num_cams):
            mse_loss += torch.nn.functional.mse_loss(result_cols[q].double(), desired_cols[q].double())

        if j % 10 == 0:
            print("STEP " + str(j) + "\t LOSS: " + str(mse_loss.item()/2))

        if use_barrier_loss:
            barrier_loss = opt.barrier_fn(H, min_val = 0.1, max_val = max_height)
            mse_loss = mse_loss + 0.02*barrier_loss

        if use_neighbor_loss:
            if paper_settings:
                neighbor_loss = opt.neighbor_fn(H)
                mse_loss = mse_loss + 0.5*neighbor_loss
            else:
                neighbor_loss = opt.neighbor_fn_wrap(H)
                mse_loss = mse_loss + neighbor_weight*neighbor_loss
    
        optimizer.zero_grad(set_to_none=False) #explicit: same behavior on all PyTorch versions (default changed in 2.0)
        mse_loss.backward()
        optimizer.step()

        if use_barrier_loss:
            #keep heights strictly inside the barrier bounds: one optimizer step can move a height past a bound,
            #which would make the barrier loss (a log) NaN for the rest of the run
            with torch.no_grad():
                H[H >= max_height] = max_height - 1e-4
                H[H <= 0.1] = 0.1 + 1e-4

    save_surface(H, COL, num_repeats, output_dir, date_time, cam_directions, image_paths, camera_z)
    repeated_H = H.repeat(num_repeats, num_repeats)
    grad_descent.last_output = os.path.join(output_dir, date_time)
    return repeated_H, COL, result_cols, desired_cols
