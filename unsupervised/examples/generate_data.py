"""
Demo render. 
1. save / load textured .obj file
2. render using SoftRas with different sigma / gamma
"""
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import torch

import soft_renderer as sr


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def generate(obj_file, renderer):
    camera_distance = 2.732
    elevation = 30

    # load from Wavefront .obj file
    try:
        mesh = sr.Mesh.from_obj(obj_file,
                                load_texture=True, texture_res=5, texture_type='surface')
    except:
        mesh = sr.Mesh.from_obj(obj_file, load_texture=False)

    orientations = []
    # draw object from different view
    for num, azimuth in enumerate(tqdm.tqdm(list(range(0, 360, 10)))):
        # rest mesh to initial state
        mesh.reset_()
        mesh.vertices *= 2
        renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        images = renderer.render_mesh(mesh)
        image = images.detach().cpu().numpy()[0]
        orientations.append((255 * image).astype(np.uint8))
        # imageio.imsave(os.path.join(output_dir, '%02d.png' % num), (255 * image).astype(np.uint8))
    return orientations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--cls', type=str,
        default='02958343')
    parser.add_argument('-b', '--id', type=str,
        default='')
    args = parser.parse_args()
    args.id = args.id.split('/')[-1]

    root = '/home/grisw/Downloads/ShapeNetCore.v2'
    output_root = os.path.join(data_dir, 'dataset_256_36_1')

    # create renderer with SoftRas
    renderer = sr.SoftRenderer(image_size=256, camera_mode='look_at')
    os.makedirs(os.path.join(output_root, args.cls), exist_ok=True)

    orientations = generate(os.path.join(root, args.cls, args.id, 'models', 'model_normalized.obj'), renderer)
    np.savez_compressed(os.path.join(output_root, args.cls, args.id), arr_0=np.array(orientations))


def compose():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--cls', type=str,
        default='02958343')
    args = parser.parse_args()

    output_root = os.path.join(data_dir, 'dataset_256_36')
    objs = []
    for f in os.listdir(os.path.join(output_root, args.cls)):
        a = np.load(os.path.join(output_root, args.cls, f))['arr_0']
        objs.append(a)
    np.savez_compressed(os.path.join(output_root, args.cls), arr_0=np.array(objs))
    print(np.array(objs).shape)


if __name__ == '__main__':
    main()
