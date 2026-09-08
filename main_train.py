from argparse import ArgumentParser
import torch
from models.trainer import *
from utils_ import str2bool
print(torch.cuda.is_available())

"""
the main function for training the CD netwo rks
"""


def train(args):
    dataloaders = utils_.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models(args=args)
    # model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils_.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(args)


if __name__ == '__main__':
    # ------------s
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='LEVIR', type=str)
    #SYSU_res18_coMDE2_AFF_Bcedice_l0_Adamw_0.0001_200
    #LEVIR_transformer_CoDEM_AFF_ce_Adamw_0.0001_200_2
    # LEVIR-CD_SEIFNet_ce_Adamw_0.0001_200
    parser.add_argument('--checkpoint_root', default='DVM_Net对比训练批次4-2', type=str)                                                                    
    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str,help='ChangeDetection|MSRSCD|LEVIR|DSFIN|SYSU-CD|LEVIR+|BBCD|WHU-CD')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--net_G', default='DVM_Net', type=str,
                        help='FC_EF | FC_Siam_conc | '
                             'FC_Siam_diff | UNet++|SNUNet|'
                             'DTCDSCN|IFNet|'
                             'base_transformer_pos_s4_dd8_dedim8|'
                             'ChangeFormer|'
                             'A2Net|DMINet|TFI-GR|SMOWNet|DVM_Net'
                            'SEIFNet')
    parser.add_argument('--backbone', default='L-Backbone', type=str, choices=['resnet', 'swin', 'vitae','L-Backbone-cross','L-Backbone','BiFormer'],
                        help='type of model')
    parser.add_argument('--mode', default='None', type=str,
                        choices=['imp','res18 ','rsp_40', 'rsp_100', 'rsp_120', 'rsp_300', 'rsp_300_sgd', 'seco','None'],
                        help='type of pretrn')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)#UNet++和A2net时为True，不需要时为False

    parser.add_argument('--loss_SD', default=False,type=str2bool) #IFNet DMINet才为True
    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--max_epochs', default=200, type=int) #150
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=200, type=int)

    args = parser.parse_args()
    utils_.get_device(args)
    print(args.gpu_ids)


    # ==============================
    # 定义要测试的 λ 值列表（升序）
    # ==============================
    # lambda_list = [0.15, 0.35, 0.55, 0.75, 0.95]
    lambda_list = [0.55]
    # ==============================
    # 多次运行配置（这里改三组实验）
    # ==============================
    run_configs = [
        # {"project_name": "LEVIR", "data_name": "LEVIR"},
        # {"project_name": "WHU-CD", "data_name": "WHU-CD"},
        {"project_name": "MSRSCD", "data_name": "MSRSCD"},
    ]

    # ==============================
    # 连续运行三次
    # ==============================
    # for run_id, cfg in enumerate(run_configs):
    #     print("\n" + "=" * 60)
    #     print(f"🚀 第 {run_id + 1} 次训练开始")
    #     print(f"📁 project_name = {cfg['project_name']}")
    #     print(f"📊 data_name = {cfg['data_name']}")
    #     print("=" * 60)

    #     # 设置当前运行参数
    #     args.project_name = cfg["project_name"]
    #     args.data_name = cfg["data_name"]

    #     # 设置保存路径
    #     run_folder = f"{args.project_name}"
    #     args.checkpoint_dir = os.path.join(args.checkpoint_root, run_folder)
    #     os.makedirs(args.checkpoint_dir, exist_ok=True)

    #     args.vis_dir = os.path.join('vis', run_folder)
    #     os.makedirs(args.vis_dir, exist_ok=True)

    #     # 可选：改变随机种子以确保每次初始化不同
    #     torch.manual_seed(42 + run_id)

    #     # ==============================
    #     # 开始训练 + 测试
    #     # ==============================
    #     train(args)
    #     test(args)

        # ==============================
        # 连续运行多个 λ 值
        # ==============================
    for run_id, cfg in enumerate(run_configs):
        for lambda_val in lambda_list:
                print("\n" + "=" * 60)
                print(f"🚀 训练: dataset={cfg['data_name']}, lambda={lambda_val}")
                print("=" * 60)

                # 设置当前运行参数
                args.data_name = cfg["data_name"]
                # 修改 project_name 以区分不同 λ（保存路径会不同）
                args.project_name = f"{cfg['project_name']}_lambda{lambda_val}"
                args.lambda_loss = lambda_val   # 将 λ 值传递给 args

                # 设置保存路径
                run_folder = f"{args.project_name}"
                args.checkpoint_dir = os.path.join(args.checkpoint_root, run_folder)
                os.makedirs(args.checkpoint_dir, exist_ok=True)

                args.vis_dir = os.path.join('vis', run_folder)
                os.makedirs(args.vis_dir, exist_ok=True)

                # 可选：改变随机种子，保证每次初始化不同（也可固定一个种子便于复现）
                torch.manual_seed(42 + run_id * len(lambda_list) + lambda_list.index(lambda_val))

                # 开始训练和测试
                train(args)
                test(args)









    # #  checkpoints dir
    # args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    # #  visualize dir
    # args.vis_dir = os.path.join('vis', args.project_name)
    # os.makedirs(args.vis_dir, exist_ok=True)

    # train(args)

    # test(args)
