"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
"encoder_decoder + new dataset"
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
    def __init__(self, directory=None, class_id=None):
        self.class_id = class_id
        self.directory = directory
        self.elevation = 30.
        self.distance = 2.732
        self.buffer = {}

        self.objs = os.listdir(os.path.join(directory, class_id))

    def get_random_batch(self, batch_size):
        images_a = []
        viewpoint_ids_a = torch.zeros(batch_size)
        for i in range(batch_size):
            object_id = np.random.choice(self.objs)
            if os.path.join(self.directory, self.class_id, object_id) in self.buffer:
                images = self.buffer[os.path.join(self.directory, self.class_id, object_id)]
            else:
                if len(self.buffer) > 500:
                    self.buffer.popitem()
                images = self.buffer[os.path.join(self.directory, self.class_id, object_id)] = np.load(os.path.join(self.directory, self.class_id, object_id))['arr_0']
            viewpoint_ids = np.random.choice(36, 2)
            images_a.append(images[viewpoint_ids[0]])

            viewpoint_ids_a[i] = int(viewpoint_ids[0])

        images_a = np.array(images_a).astype('float32') / 255.

        distances = torch.ones(batch_size) * self.distance
        elevations = torch.ones(batch_size) * self.elevation

        return torch.from_numpy(images_a), viewpoint_ids_a * 10, distances, elevations

    def get_val(self, object_id, view):
        images = np.load(os.path.join(self.directory, self.class_id, self.objs[object_id]))['arr_0']
        image = np.array([images[view].astype('float32') / 255.])
        return torch.from_numpy(image)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2),
            nn.ReLU()
        )

        # self.fc1 = nn.Sequential(
        #     nn.Linear(15*15*128, 1024),
        #     nn.ReLU()
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, 2048),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(2048, 1024),
        #     nn.ReLU()
        # )

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.fc1(x.view(-1, 15*15*256))
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

"""
Decoder without deConv
"""
class Decoder(nn.Module):
    def __init__(self, template_path):
        super(Decoder, self).__init__()

        self.template_mesh = sr.Mesh.from_obj(template_path)
        self.register_buffer('vertices', self.template_mesh.vertices * 0.55)
        self.register_buffer('faces', self.template_mesh.faces)

        self.laplacian_loss = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())

        self.fc1 = nn.Sequential(
            nn.Linear(15*15*256, 5000),
            nn.ReLU()
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.ReLU()
        # )
        self.fc_center = nn.Linear(5000, 3)
        self.fc_displace = nn.Linear(5000, self.template_mesh.num_vertices * 3)

    def forward(self, inputs):
        x = inputs.view(-1, 15*15*256)
        x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        base = torch.log(self.vertices.abs() / (1 - self.vertices.abs())).repeat(inputs.size(0), 1, 1)
        centroid = torch.tanh(self.fc_center(x).view(-1, 1, 3)).repeat(1, self.template_mesh.num_vertices, 1)
        displace = self.fc_displace(x).view(-1, self.template_mesh.num_vertices, 3)
        vertices = torch.sigmoid(base + displace) * torch.sign(self.vertices)
        vertices = F.relu(vertices) * (1 - centroid) - F.relu(-vertices) * (centroid + 1)
        vertices = vertices + centroid

        # vertices = torch.tanh(self.fc_displace(x)).view(-1, self.template_mesh.num_vertices, 3)

        # apply Laplacian and flatten geometry constraints
        laplacian_loss = self.laplacian_loss(vertices).mean()
        flatten_loss = self.flatten_loss(vertices).mean()

        return sr.Mesh(vertices, self.faces.repeat(inputs.size(0), 1, 1)), laplacian_loss, flatten_loss


"""
Decoder with deConv
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
#         self.fc2 = nn.Sequential(
#             nn.Linear(23*23*64, 2048),
#             nn.ReLU())
#
#         self.fc_center = nn.Linear(2048, 3)
#
#         self.fc_displace = nn.Linear(2048, self.template_mesh.num_vertices * 3)
#
#     def forward(self, inputs):
#         x = self.fc1(inputs)
#         x = x.view(-1, 512, 2, 2)
#         x = self.deconv1(x)
#         x = self.deconv2(x)
#         x = self.deconv3(x)
#         x = x.view(-1, 23*23*64)
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


def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union).sum() / intersect.nelement()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--classes', type=str,
                        default='02958343')  # ,02691156
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = nn.Sequential(
        Encoder(),
        Decoder(args.template_mesh)
    ).cuda()

    if args.load:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'v6-%d.pth' % args.train_num)))

    renderer = sr.SoftRenderer(image_size=256, sigma_val=1e-4, aggr_func_rgb='hard', camera_mode='look_at')

    # read training images and camera poses
    dataset_train = ShapeNet(os.path.join(data_dir, 'dataset_256_36_1'), args.classes)
    optimizer = torch.optim.Adam(model.parameters())

    if not args.load:
        writer = imageio.get_writer(os.path.join(args.output_dir, 'train.gif'), mode='I')
        train_num = args.train_num
        loop = tqdm.tqdm(list(range(0, train_num)))

        Loss_list = np.zeros(train_num)
        for i in loop:
            images_a, viewpoints_a, distances, elevations = dataset_train.get_random_batch(args.batch_size)
            images_gt = images_a.cuda()

            mesh, laplacian_loss, flatten_loss = model(images_gt)
            renderer.transform.set_eyes_from_angles(distances, elevations, viewpoints_a)
            images_pred = renderer.render_mesh(mesh)

            # optimize mesh with silhouette reprojection error and
            # geometry constraints
            loss = neg_iou_loss(images_pred[:, 3], images_gt[:, 3]) + \
                   0.07 * laplacian_loss + \
                   0.004 * flatten_loss

            Loss_list[i] = loss.item()
            loop.set_description('Loss: %.4f'%(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                image = images_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
                imagegt = images_gt.detach().cpu().numpy()[0].transpose((1, 2, 0))
                writer.append_data((255 * image).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'deform_%05d.png'%i), (255*image[..., -1]).astype(np.uint8))
                imageio.imsave(os.path.join(args.output_dir, 'gt_%05d.png' % i), (255 * imagegt[..., -1]).astype(np.uint8))

        torch.save(model.state_dict(), os.path.join(args.output_dir, 'v6-%d.pth' % args.train_num))

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

    # save optimized mesh
    vals = torch.cat((dataset_train.get_val(0, 4),
                     dataset_train.get_val(1, 16),
                     dataset_train.get_val(2, 8),
                     dataset_train.get_val(3, 20))).cuda()

    val_res = model(vals)[0]
    for i in range(vals.shape[0]):
        val_res.save_obj(os.path.join(args.output_dir, 'car_%d.obj' % i), idx=i, save_texture=False)
        img = vals[i].detach().cpu().numpy().transpose((1, 2, 0))
        imageio.imsave(os.path.join(args.output_dir, 'car_%d.png' % i), (255 * img).astype(np.uint8))

    "Plane dataset"
    # model(dataset_val.get_val(args.classes.split(',')[1], 0, 4).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_1.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 1, 16).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_2.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 2, 8).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_3.obj'), save_texture=False)
    # model(dataset_val.get_val(args.classes.split(',')[1], 3, 20).cuda())[0].save_obj(os.path.join(args.output_dir, 'plane_4.obj'), save_texture=False)
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
