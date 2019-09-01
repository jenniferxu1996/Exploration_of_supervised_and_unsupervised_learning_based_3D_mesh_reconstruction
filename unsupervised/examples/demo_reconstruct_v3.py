"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
"encoder_decoder new version"
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
import resnet

import soft_renderer as sr

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            # (2831, 24, 4, 64, 64)
            images.append(np.load(
                os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name)))['arr_0'])
            voxels.append(np.load(
                os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name)))['arr_0'])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 64, 64))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        del images
        del voxels

    def get_val(self, class_id, object_id, view):
        data_id = (object_id + self.pos[class_id]) * 24 + view
        image = self.images[[data_id]].astype('float32') / 255.
        return torch.from_numpy(image)

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])
            viewpoint_id_a = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_ids_a[i] = data_id_a
            viewpoint_ids_a[i] = viewpoint_id_a

        images_a = self.images[data_ids_a].astype('float32') / 255.

        distances = torch.ones(batch_size) * self.distance
        elevations = torch.ones(batch_size) * self.elevation

        return torch.from_numpy(images_a), distances, elevations, -viewpoint_ids_a * 15

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids
        for i in range((data_ids.size - 1) / batch_size + 1):
            images = self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.
            voxels = self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] / 24]
            yield images, voxels


"""
Decoder without deConv
"""
# class Decoder(nn.Module):
#     def __init__(self, template_path):
#         super(Decoder, self).__init__()
#
#         self.template_mesh = sr.Mesh.from_obj(template_path)
#         self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
#         self.register_buffer('faces', self.template_mesh.faces)
#
#         self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
#         self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
#
#         self.fc1 = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(1024, 2048),
#             nn.ReLU()
#         )
#         self.fc_center = nn.Linear(2048, 3)
#         self.fc_displace = nn.Linear(2048, self.template_mesh.num_vertices * 3)
#
#     def forward(self, inputs):
#         x = self.fc1(inputs)
#         x = self.fc2(x)
#
#         base = torch.log(self.vertices.abs() / (1 - self.vertices.abs())).repeat(inputs.size(0), 1, 1)
#         centroid = torch.tanh(self.fc_center(x).view(-1, 1, 3)).repeat(1, self.template_mesh.num_vertices, 1)
#         displace = self.fc_displace(x).view(-1, self.template_mesh.num_vertices, 3)
#         vertices = torch.sigmoid(base + displace) * torch.sign(self.vertices)
#         vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
#         vertices = vertices + centroid
#
#         # apply Laplacian and flatten geometry constraints
#         laplacian_loss = self.laplacian_loss(vertices).mean()
#         flatten_loss = self.flatten_loss(vertices).mean()
#
#         return sr.Mesh(vertices, self.faces.repeat(inputs.size(0), 1, 1)), laplacian_loss, flatten_loss


"""
Decoder with deConv
"""
class Decoder(nn.Module):
    def __init__(self, template_path):
        super(Decoder, self).__init__()

        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.5)
        self.register_buffer('faces', self.template_mesh.faces)

        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

        self.fc1 = nn.Sequential(
            nn.Linear(512, 2*2*512),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(23*23*64, 2048),
            nn.ReLU())

        self.fc_center = nn.Linear(2048, 3)

        self.fc_displace = nn.Linear(2048, self.template_mesh.num_vertices * 3)

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = x.view(-1, 512, 2, 2)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = x.view(-1, 23*23*64)
        x = self.fc2(x)

        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs())).repeat(inputs.size(0), 1, 1)
        centroid = torch.tanh(self.fc_center(x).view(-1, 1, 3)).repeat(1, self.template_mesh.num_vertices, 1)
        displace = self.fc_displace(x).view(-1, self.template_mesh.num_vertices, 3)
        vertices = torch.sigmoid(base + displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices, self.faces.repeat(inputs.size(0), 1, 1)), laplacian_loss, flatten_loss
#

def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def main(_args=None):
    if _args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--classes', type=str,
                            default='02958343,04256520')  # ,
        parser.add_argument('-l', '--load', type=bool,
                            default=False)
        parser.add_argument('-t', '--template-mesh', type=str,
                            default=os.path.join(data_dir, 'obj/sphere/sphere_1352.obj'))
        parser.add_argument('-o', '--output-dir', type=str,
                            default=os.path.join(data_dir, 'results/output_reconstruct'))
        parser.add_argument('-b', '--batch-size', type=int,
                            default=64)
        parser.add_argument('-n', '--train-num', type=int,
                            default=5000)
        parser.add_argument('-i', '--image', type=str)
        args = parser.parse_args()
    else:
        args = _args

    os.makedirs(args.output_dir, exist_ok=True)

    model = nn.Sequential(
        resnet.resnet18(num_classes=512),
        Decoder(args.template_mesh)
    ).cuda()

    if args.load:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'v3-%d.pth' % args.train_num)))

    renderer = sr.SoftRenderer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard',
                               camera_mode='look_at', viewing_angle=15)

    # read training images and camera poses
    dataset_train = ShapeNet(os.path.join(data_dir, 'dataset'), args.classes.split(','), 'train')
    dataset_val = ShapeNet(os.path.join(data_dir, 'dataset'), args.classes.split(','), 'val')
    optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.9, 0.999))

    if not args.load:
        writer = imageio.get_writer(os.path.join(args.output_dir, 'train.gif'), mode='I')
        train_num = args.train_num
        loop = tqdm.tqdm(list(range(0, train_num)))

        Loss_list = np.zeros(train_num)

        for i in loop:
            images, distances, elevations, viewpoints = dataset_train.get_random_batch(args.batch_size)
            images_gt = images.cuda()

            mesh, laplacian_loss, flatten_loss = model(images_gt)
            renderer.transform.set_eyes_from_angles(distances, elevations, viewpoints)
            images_pred = renderer.render_mesh(mesh)

            # optimize mesh with silhouette reprojection error and
            # geometry constraints
            loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3]) + \
                   0.05 * laplacian_loss + \
                   0.001 * flatten_loss

            Loss_list[i]=loss
            loop.set_description('Loss: %.4f'%(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                image = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writer.append_data((255 * image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image[..., -1]).astype(np.uint8))

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'v3-%d.pth' % args.train_num))

        fig = plt.figure()
        x = range(0, train_num)
        y = Loss_list
        plt.plot(x, y)
        plt.xlabel('Iteration')
        plt.ylabel('Test loss')
        plt.show()
        fig.savefig("loss.jpg")
        np.set_printoptions(threshold=np.inf)
        print(Loss_list)
        # print(Loss_list)

    val_iou_loss = 0
    for i in tqdm.tqdm(range(0, dataset_val.images.shape[0], 24)):
        val_imgs = torch.from_numpy(dataset_val.images[i:i+24].astype('float32') / 255.)
        val_imgs = val_imgs.cuda()
        val_distances = torch.ones(val_imgs.size(0)) * dataset_val.distance
        val_elevations = torch.ones(val_imgs.size(0)) * dataset_val.elevation
        val_viewpoint_ids = torch.ones(24)
        for v in range(24):
            val_viewpoint_ids[v] = v

        val_mesh, val_laplacian_loss, val_flatten_loss = model(val_imgs)
        renderer.transform.set_eyes_from_angles(val_distances, val_elevations, -val_viewpoint_ids * 15)
        val_images_pred = renderer.render_mesh(val_mesh)
        val_iou_loss += neg_iou_loss(val_images_pred[:, 3], val_imgs[:, 3]).item()
    print('Val IOU loss: ', val_iou_loss / (dataset_val.images.shape[0] / 24))

    if args.image:
        img = imageio.imread(args.image)
        model(torch.from_numpy(np.array([img.astype('float32') / 255.]).reshape((-1, 4, 64, 64))).cuda())[0].save_obj(os.path.join(args.output_dir, 'a.obj'), save_texture=False)

    # save optimized mesh
    model(dataset_val.get_val(args.classes.split(',')[0], 0, 4).cuda())[0].save_obj(os.path.join(args.output_dir, '1.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 1, 16).cuda())[0].save_obj(os.path.join(args.output_dir, '2.obj'), save_texture=False)

    model(dataset_val.get_val(args.classes.split(',')[0], 2, 8).cuda())[0].save_obj(os.path.join(args.output_dir, '3.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 10).cuda())[0].save_obj(os.path.join(args.output_dir, '3_0.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 12).cuda())[0].save_obj(os.path.join(args.output_dir, '3_1.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 1).cuda())[0].save_obj(os.path.join(args.output_dir, '3_2.obj'), save_texture=False)

    model(dataset_val.get_val(args.classes.split(',')[0], 3, 20).cuda())[0].save_obj(os.path.join(args.output_dir, '4.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 3, 1).cuda())[0].save_obj(os.path.join(args.output_dir, '4_0.obj'), save_texture=False)

    img3_0 = dataset_val.get_val(args.classes.split(',')[0], 2, 10).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img3_1 = dataset_val.get_val(args.classes.split(',')[0], 2, 12).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img3_2 = dataset_val.get_val(args.classes.split(',')[0], 2, 1).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img1 = dataset_val.get_val(args.classes.split(',')[0], 0, 4).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img2 = dataset_val.get_val(args.classes.split(',')[0], 1, 16).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img3 = dataset_val.get_val(args.classes.split(',')[0], 2, 8).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img4 = dataset_val.get_val(args.classes.split(',')[0], 3, 20).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # imageio.imsave(os.path.join(args.output_dir, '233.png'), (255 * img233[..., -1]).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, '235.png'), (255 * img235[..., -1]).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, 'car_1.png'), (255 * img1).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, 'car_2.png'), (255 * img2).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'car_3.png'), (255 * img3).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'car_4.png'), (255 * img4).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'car_3_0.png'), (255 * img3_0).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'car_3_1.png'), (255 * img3_1).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'car_3_2.png'), (255 * img3_2).astype(np.uint8))

    "Plane dataset"
    model(dataset_val.get_val(args.classes.split(',')[1], 0, 4).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_1.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[1], 1, 16).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_2.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[1], 2, 8).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_3.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[1], 3, 20).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_4.obj'), save_texture=False)
    img1 = dataset_val.get_val(args.classes.split(',')[1], 0, 4).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img2 = dataset_val.get_val(args.classes.split(',')[1], 1, 16).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img3 = dataset_val.get_val(args.classes.split(',')[1], 2, 8).detach().cpu().numpy()[0].transpose((1, 2, 0))
    img4 = dataset_val.get_val(args.classes.split(',')[1], 3, 20).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # imageio.imsave(os.path.join(args.output_dir, '233.png'), (255 * img233[..., -1]).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, '235.png'), (255 * img235[..., -1]).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'plane_1.png'), (255 * img1).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'plane_2.png'), (255 * img2).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'plane_3.png'), (255 * img3).astype(np.uint8))
    imageio.imsave(os.path.join(args.output_dir, 'plane_4.png'), (255 * img4).astype(np.uint8))



if __name__ == '__main__':
    main()
