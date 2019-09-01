import os
import argparse
import demo_reconstruct_v1
import demo_reconstruct_v2
import demo_reconstruct_v3
import demo_reconstruct_v4

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, '../data')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=int,
                        default=1)
    parser.add_argument('-c', '--classes', type=str,
                        default='02958343')
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

    if args.experiment == 1:
        demo_reconstruct_v1.main(args)
    elif args.experiment == 2:
        demo_reconstruct_v2.main(args)
    elif args.experiment == 3:
        demo_reconstruct_v3.main(args)
    elif args.experiment == 4:
        demo_reconstruct_v4.main(args)
    else:
        print('Unknown experiment.')


if __name__ == '__main__':
    main()
