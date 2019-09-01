"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
"discriminator added"
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import imageio
import argparse
from spectral_norm import SpectralNorm
import resnet
from torch.autograd import Function

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

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])
            viewpoint_ids = np.random.choice(24, 2)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_ids[0]
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_ids[1]
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = int(viewpoint_ids[0])
            viewpoint_ids_b[i] = int(viewpoint_ids[1])

        images_a = self.images[data_ids_a].astype('float32') / 255.
        images_b = self.images[data_ids_b].astype('float32') / 255.

        distances = torch.ones(batch_size) * self.distance
        elevations = torch.ones(batch_size) * self.elevation

        return torch.from_numpy(images_a), torch.from_numpy(images_b), -viewpoint_ids_a * 15, -viewpoint_ids_b * 15, distances, elevations

    def get_val(self, class_id, object_id, view):
        data_id = (object_id + self.pos[class_id]) * 24 + view
        image = self.images[[data_id]].astype('float32') / 255.
        return torch.from_numpy(image)


    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids
        for i in range((data_ids.size - 1) / batch_size + 1):
            images = self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.
            voxels = self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] / 24]
            yield images, voxels


class Discriminator(nn.Module):
    def __init__(self, lambd):
        super(Discriminator, self).__init__()

        self.grl = GradReverse(lambd)
        self.image_conv = nn.Sequential(
            SpectralNorm(nn.Conv2d(4, 32, 3, stride=2)),
            nn.LeakyReLU()
        )
        self.viewpoint_fc = nn.Sequential(
            SpectralNorm(nn.Linear(3, 32)),
            nn.LeakyReLU()
        )
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 64, 3, stride=2)),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(64, 128, 3, stride=2)),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 256, 3, stride=2)),
            nn.LeakyReLU()
        )
        # self.conv4 = nn.Sequential(
        #     SpectralNorm(nn.Conv2d(256, 256, 2, stride=2)),
        #     nn.LeakyReLU()
        # )
        # self.conv5 = SpectralNorm(nn.Conv2d(256, 1, 2, stride=2))
        self.fc = nn.Sequential(
            SpectralNorm(nn.Linear(4 * 4 * 256, 1)),
            nn.Sigmoid()
        )

    def forward(self, x1, v1):
        """
        :param x1: input image
        :param v1: input viewpoint
        :param x2: randomly generated image
        :param v2: randomly generated viewpoint
        :return:
        """
        x1 = self.image_conv(x1)
        v1 = self.viewpoint_fc(v1)
        v1 = v1.unsqueeze(2).unsqueeze(2).repeat(1, 1, 31, 31)
        x1 = torch.cat((x1, v1), 1)
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        # x1 = self.conv4(x1)
        # x1 = self.conv5(x1)
        x1 = self.grl(x1)
        x1 = self.fc(x1.view(-1, 4*4*256))

        return x1


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
#             nn.Linear(512, 2*2*512),
#             nn.ReLU()
#         )
#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(512, 256, 3, stride=2),
#             nn.ReLU()
#         )
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 3, stride=2),
#             nn.ReLU()
#         )
#         self.deconv3 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 3, stride=2),
#             nn.ReLU()
#         )
#         self.fc2 = nn.Linear(23*23*64, self.template_mesh.num_vertices * 3)
#
#     def forward(self, inputs):
#         x = self.fc1(inputs)
#         x = x.view(-1, 512, 2, 2)
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = self.deconv3(x)
#         x = x.view(-1, 23*23*64)
#         x = self.fc2(x)
#         x = x.view(-1, self.template_mesh.num_vertices, 3)
#
#         # apply Laplacian and flatten geometry constraints
#         laplacian_loss = self.laplacian_loss(x).mean()
#         flatten_loss = self.flatten_loss(x).mean()
#
#         return sr.Mesh(x, self.faces.repeat(x.size(0), 1, 1)), laplacian_loss, flatten_loss
#

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


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def main(_args=None):
    if _args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--classes', type=str,
                            default='02958343') #,02691156
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
        parser.add_argument('-v', '--random-views', type=int,
                            default=1)
        args = parser.parse_args()
    else:
        args = _args

    os.makedirs(args.output_dir, exist_ok=True)

    model = nn.Sequential(
        resnet.resnet18(num_classes=512),
        Decoder(args.template_mesh)
    ).cuda()

    if args.load:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'v4_mesh_gen-%d.pth' % args.train_num)))

    dis = Discriminator(0.2).cuda()

    if args.load:
        dis.load_state_dict(torch.load(os.path.join(args.output_dir, 'v4_dis-%d.pth' % args.train_num)))

    renderer = sr.SoftRenderer(image_size=64, sigma_val=1e-4, aggr_func_rgb='hard',
                               camera_mode='look_at', viewing_angle=15)

    # read training images and camera poses
    dataset_train = ShapeNet(os.path.join(data_dir, 'dataset'), args.classes.split(','), 'train')
    dataset_val = ShapeNet(os.path.join(data_dir, 'dataset'), args.classes.split(','), 'val')
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': dis.parameters()}], 0.0001, betas=(0.9, 0.999))

    if not args.load:
        writer = imageio.get_writer(os.path.join(args.output_dir, 'train.gif'), mode='I')

        train_num = args.train_num
        loop = tqdm.tqdm(list(range(0, train_num)))
        Loss_list = np.zeros(train_num)
        Loss_ld = np.zeros(train_num)

        for i in loop:
            images_a, images_b, viewpoints_a, viewpoints_b, distances, elevations = dataset_train.get_random_batch(args.batch_size)
            images_gt_a = images_a.cuda()

            mesh, laplacian_loss, flatten_loss = model(images_gt_a)
            renderer.transform.set_eyes_from_angles(distances, elevations, viewpoints_a)
            images_pred_a = renderer.render_mesh(mesh)

            va = torch.cat((distances.unsqueeze(1), elevations.unsqueeze(1), viewpoints_a.unsqueeze(1)), 1).cuda()
            p_a = dis(images_pred_a, va)

            random_pbs = 0
            for v in range(args.random_views):
                random_view = torch.zeros(args.batch_size)
                for b in range(args.batch_size):
                    random_view[b] = np.random.choice(24)
                mesh.reset_()
                renderer.transform.set_eyes_from_angles(distances, elevations, random_view)
                images_pred_b = renderer.render_mesh(mesh)

                vb = torch.cat((distances.unsqueeze(1), elevations.unsqueeze(1), random_view.unsqueeze(1)), 1).cuda()
                p_b = dis(images_pred_b, vb)
                random_pbs += torch.log(1-p_b).mean()

            ld = -torch.log(p_a).mean() - random_pbs / args.random_views

            # optimize mesh with silhouette reprojection error and
            # geometry constraints
            lossIoU1 = neg_iou_loss(images_pred_a[:, 3], images_gt_a[:, 3])

            loss = lossIoU1 + ld + \
                   0.05 * laplacian_loss + \
                   0.001 * flatten_loss
            Loss_ld[i] = ld.item()
            Loss_list[i] = (loss - ld).item()

            loop.set_description('Loss: %.4f'%(loss.item()))
            print('loss:', Loss_list[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                image = images_pred_a.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writer.append_data((255 * image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image[..., -1]).astype(np.uint8))
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'v4_mesh_gen-%d.pth' % args.train_num))
        torch.save(dis.state_dict(), os.path.join(args.output_dir, 'v4_dis-%d.pth' % args.train_num))
        fig = plt.figure()
        x = range(0, train_num)
        y = Loss_list
        plt.plot(x, y)
        plt.xlabel('Iteration')
        plt.ylabel('Test loss')
        plt.show()
        fig.savefig("loss_r_d.jpg")
        np.set_printoptions(threshold=np.inf)
        print(Loss_list)
        print(Loss_ld)

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

    # save optimized mesh
    # model(dataset_val.get_val(args.classes, 0, 4).cuda())[0].save_obj(os.path.join(args.output_dir, '1.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes, 1, 16).cuda())[0].save_obj(os.path.join(args.output_dir, '2.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes, 2, 8).cuda())[0].save_obj(os.path.join(args.output_dir, '3.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes, 3, 20).cuda())[0].save_obj(os.path.join(args.output_dir, '4.obj'), save_texture=False)
    #
    # img1 = dataset_val.get_val(args.classes, 0, 4).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img2 = dataset_val.get_val(args.classes, 1, 16).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img3 = dataset_val.get_val(args.classes, 2, 8).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img4 = dataset_val.get_val(args.classes, 3, 20).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # # imageio.imsave(os.path.join(args.output_dir, '233.png'), (255 * img233[..., -1]).astype(np.uint8))
    # # imageio.imsave(os.path.join(args.output_dir, '235.png'), (255 * img235[..., -1]).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, '1.png'), (255 * img1).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, '2.png'), (255 * img2).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, '3.png'), (255 * img3).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, '4.png'), (255 * img4).astype(np.uint8))

    # model(dataset_val.get_val(args.classes.split(',')[0], 0, 4).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_1.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[0], 1, 16).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_2.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 8).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_3.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 10).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_3_0.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 12).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_3_1.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 2, 1).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_3_2.obj'), save_texture=False)

    model(dataset_val.get_val(args.classes.split(',')[0], 3, 20).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_4.obj'), save_texture=False)
    model(dataset_val.get_val(args.classes.split(',')[0], 3, 1).cuda())[0].save_obj(os.path.join(args.output_dir, 'car_4_0.obj'), save_texture=False)

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
    # model(dataset_val.get_val(args.classes.split(',')[1], 0, 4).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_1.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 1, 16).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_2.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 2, 8).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_3.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 2, 10).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_3_0.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 3, 20).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_4.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 3, 1).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_4_0.obj'), save_texture=False)
    #
    # img1 = dataset_val.get_val(args.classes.split(',')[1], 0, 4).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img2 = dataset_val.get_val(args.classes.split(',')[1], 1, 16).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img3 = dataset_val.get_val(args.classes.split(',')[1], 2, 8).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # img4 = dataset_val.get_val(args.classes.split(',')[1], 3, 20).detach().cpu().numpy()[0].transpose((1, 2, 0))
    # # imageio.imsave(os.path.join(args.output_dir, '233.png'), (255 * img233[..., -1]).astype(np.uint8))
    # # imageio.imsave(os.path.join(args.output_dir, '235.png'), (255 * img235[..., -1]).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, 'plane_1.png'), (255 * img1).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, 'plane_2.png'), (255 * img2).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, 'plane_3.png'), (255 * img3).astype(np.uint8))
    # imageio.imsave(os.path.join(args.output_dir, 'plane_4.png'), (255 * img4).astype(np.uint8))
if __name__ == '__main__':
    main()
