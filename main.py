import logging
import os
import argparse
import sys
import json
import wandb
import dotenv

import torch
import copy

# Load R2 credentials from .env
dotenv.load_dotenv()

def upload_checkpoint_to_r2(local_path, cloud_name):
    import os
    import boto3
    
    access_key = os.getenv("CF_R2_ACCESS_KEY_ID")
    secret_key = os.getenv("CF_R2_SECRET_ACCESS_KEY")
    endpoint = os.getenv("CF_R2_ENDPOINT_URL")
    bucket_name = os.getenv("CF_R2_BUCKET_NAME")
    
    if not all([access_key, secret_key, endpoint, bucket_name]) or "your_" in access_key:
        print("Cloudflare R2 credentials are not configured in .env. Skipping cloud upload.")
        return
    
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint
        )
        print(f"Uploading {local_path} to R2 bucket {bucket_name} as {cloud_name}...")
        s3.upload_file(local_path, bucket_name, cloud_name)
        print("Upload completed successfully!")
    except Exception as e:
        print(f"Failed to upload checkpoint to R2: {e}")

def recover_checkpoint_from_r2(local_dir, cloud_name):
    import os
    import boto3
    
    access_key = os.getenv("CF_R2_ACCESS_KEY_ID")
    secret_key = os.getenv("CF_R2_SECRET_ACCESS_KEY")
    endpoint = os.getenv("CF_R2_ENDPOINT_URL")
    bucket_name = os.getenv("CF_R2_BUCKET_NAME")
    
    if not all([access_key, secret_key, endpoint, bucket_name]) or "your_" in access_key:
        print("Cloudflare R2 credentials are not configured in .env. Skipping cloud recovery.")
        return False
        
    local_path = os.path.join(local_dir, 'checkpoint_latest.pth')
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint
        )
        print(f"Checking cloud checkpoint {cloud_name} in bucket {bucket_name}...")
        s3.head_object(Bucket=bucket_name, Key=cloud_name)
        print(f"Downloading cloud checkpoint {cloud_name} to {local_path}...")
        os.makedirs(local_dir, exist_ok=True)
        s3.download_file(bucket_name, cloud_name, local_path)
        print("Checkpoint downloaded successfully!")
        return True
    except Exception as e:
        print(f"No cloud checkpoint found or download failed: {e}")
        return False


from algorithms.waft import WAFT
from bridgedepth.config import export_model_config
from bridgedepth.dataloader import build_train_loader
from bridgedepth.loss import build_criterion
from bridgedepth.utils import misc
import bridgedepth.utils.dist_utils as comm
from bridgedepth.utils.logger import setup_logger
from bridgedepth.utils.launch import launch
from bridgedepth.utils.eval_disp import eval_disp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_args_parser():
    parser = argparse.ArgumentParser(
        f"""
        Examples:

        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8

        Change some config options:
            $ {sys.argv[0]} SOLVER.IMS_PER_BATCH 8

        Run on multiple machines:
            (machine 0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine 1)$ {sys.argv[1]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action='store_true')
    parser.add_argument("--ckpt", default=None, help='path to the checkpoint file or model name when eval_only is True')
    parser.add_argument("--checkpoint-dir", default=None, help='path to the checkpoint directory')
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    # distributed training
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details."
    )
    parser.add_argument(
        "opts",
        help="""
        Modify config options at the end of the command. For Yacs configs, use
        space-separated "PATH.KEY VALUE" pair.
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER
    )

    return parser

def build_optimizer(model, cfg):
    base_lr = cfg.SOLVER.BASE_LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-5, eps=1e-8)
    return optimizer


def _setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the bridgedepth logger
    2. Log basic information about environment, cmdline arguments, git commit, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    data_name = args.config_file.replace('\\', '/').split('/')[-2]
    alg_name = args.config_file.replace('\\', '/').split('/')[-1].split('.')[0]
    if getattr(args, 'checkpoint_dir', None) is None:
        args.checkpoint_dir = f"ckpts/{data_name}/{alg_name}/{args.seed}"
    checkpoint_dir = args.checkpoint_dir
    if comm.is_main_process() and checkpoint_dir:
        misc.check_path(checkpoint_dir)

    rank = comm.get_rank()
    logger = setup_logger(checkpoint_dir, distributed_rank=rank, name='waft')

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + misc.collect_env_info())

    logger.info("git:\n {}\n".format(misc.get_sha()))
    logger.info("Command line arguments: " + str(args))

    if comm.is_main_process() and checkpoint_dir:
        path = os.path.join(checkpoint_dir, "config.yaml")
        with open(path, 'w') as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    misc.seed_all_rng(None if args.seed < 0 else args.seed + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def setup(args):
    """
    Create config and perform basic setups.
    """
    from bridgedepth.config import get_cfg
    cfg = get_cfg()
    if len(args.config_file) > 0:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    _setup(cfg, args)
    comm.setup_for_distributed(comm.is_main_process())
    return cfg

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def macs_profiler(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1, 3, 544, 960).to(device)
    sample = {
        "img1": input,
        "img2": input,
    }
    with torch.no_grad():
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            with_flops=True) as prof:
                output = model(sample)
    
    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cuda_time_total' if torch.cuda.is_available() else 'self_cpu_time_total', row_limit=5))
    events = prof.events()
    forward_MACs = sum([int(evt.flops) for evt in events])
    print("forward MACs: ", forward_MACs / 2 / 1e9, "G")
    print("Number of parameters: ", count_parameters(model) / 1e6, "M")


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WAFT(cfg)
    model = model.to(device)
    if device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if comm.get_world_size() > 1:
        if device.type == "cuda":
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                find_unused_parameters=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
            )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # evaluate
    if args.eval_only:
        test_model = copy.deepcopy(model)
        macs_profiler(test_model)
        print('Load checkpoint: %s' % args.ckpt)
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(weights, strict=False)
        eval_disp(model, cfg, output_dir=getattr(args, 'checkpoint_dir', 'ckpts'))
        return

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    logger = logging.getLogger("waft")
    optimizer = build_optimizer(model_without_ddp, cfg)
    criterion = build_criterion(cfg)

    # Recover checkpoint from cloud
    data_name = args.config_file.replace('\\', '/').split('/')[-2]
    alg_name = args.config_file.replace('\\', '/').split('/')[-1].split('.')[0]
    cloud_latest_name = f"{data_name}_{alg_name}_latest.pth"
    cloud_best_name = f"{data_name}_{alg_name}_best.pth"
    
    if comm.is_main_process():
        recover_checkpoint_from_r2(args.checkpoint_dir, cloud_latest_name)
    comm.synchronize()
    
    local_latest_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
    if os.path.exists(local_latest_path):
        cfg.defrost()
        cfg.SOLVER.RESUME = local_latest_path
        cfg.freeze()

    # resume checkpoints
    start_epoch = 0
    start_step = 0
    resume = cfg.SOLVER.RESUME
    no_resume_optimizer = cfg.SOLVER.NO_RESUME_OPTIMIZER
    if resume:
        logger.info('Load checkpoint: %s' % resume)
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_without_ddp.load_state_dict(weights, strict=False)
        if 'optimizer' in checkpoint and 'step' in checkpoint and 'epoch' in checkpoint and not no_resume_optimizer:
            logger.info('Load optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']

    # training dataset
    train_loader, train_sampler = build_train_loader(cfg)

    # training scheduler
    last_epoch = start_step if resume and start_step > 0 else -1
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, cfg.SOLVER.BASE_LR,
        cfg.SOLVER.MAX_ITER + 100,
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='linear',
        # anneal_strategy='cos',
        last_epoch=last_epoch
    )

    if comm.is_main_process():
        avg_dict = {}
        data_name = args.config_file.split('/')[-2]
        exp_name = args.config_file.split('/')[-1].split('.')[0] + f"-{args.seed}"
        wandb.init(
            project=data_name,
            name=exp_name,
        )

    total_steps = start_step
    epoch = start_epoch
    logger.info('Start training')

    best_epe = float('inf')
    best_checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
    has_new_best = False
    
    # Try loading best EPE from existing checkpoint if resuming
    if resume and os.path.exists(best_checkpoint_path):
        try:
            best_chk = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
            if 'epe' in best_chk:
                best_epe = best_chk['epe']
                logger.info(f"Loaded existing best EPE from checkpoint: {best_epe:.4f}")
        except Exception as e:
            pass

    print_freq = 20
    try:
        while total_steps < cfg.SOLVER.MAX_ITER:
            model.train()
        # manual change random seed for shuffling every epoch
        if comm.get_world_size() > 1:
            train_sampler.set_epoch(epoch)
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)

        header = 'Epoch: [{}]'.format(epoch)
        for i_batch, sample in enumerate(train_loader):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sample = {k: v.to(device) if hasattr(v, "to") else v for k, v in sample.items()}

            # use bf16 for mix-precision training
            autocast_enabled = cfg.SOLVER.MIX_PRECISION if device.type == "cuda" else False
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                result_dict = model(sample)
                loss_dict, metrics = criterion(result_dict, sample, log=True)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # more efficient zero_grad
            for param in model_without_ddp.parameters():
                param.grad = None

            losses.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.GRAD_CLIP)

            optimizer.step()
            lr_scheduler.step()

            if comm.is_main_process():
                if sample["valid"].sum() > 0:
                    for k, v in metrics.items():
                        avg_params = avg_dict.get(k, AverageMeter())
                        avg_params.update(v)
                        avg_dict[k] = avg_params

            total_steps += 1

            if total_steps % 100 == 0:
                if comm.is_main_process():
                    wandb_log_dict = {f"train/{k}": v.avg for k, v in avg_dict.items()}
                    wandb.log(wandb_log_dict)
                    avg_dict = {}

            # Upload best checkpoint every 50 steps if it has changed
            if total_steps % 50 == 0:
                if comm.is_main_process() and has_new_best:
                    upload_checkpoint_to_r2(best_checkpoint_path, cloud_best_name)
                    has_new_best = False

            if total_steps % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or total_steps == cfg.SOLVER.MAX_ITER:
                if comm.is_main_process():
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'model_config': export_model_config(cfg),
                    }, checkpoint_path)
                    upload_checkpoint_to_r2(checkpoint_path, f"{data_name}_{alg_name}_step_%06d.pth" % total_steps)

            if total_steps % cfg.SOLVER.LATEST_CHECKPOINT_PERIOD == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_latest.pth')
                if comm.is_main_process():
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'model_config': export_model_config(cfg),
                        'optimizer': optimizer.state_dict(),
                        'step': total_steps,
                        'epoch': epoch,
                    }, checkpoint_path)
                    upload_checkpoint_to_r2(checkpoint_path, cloud_latest_name)

            if cfg.TEST.EVAL_PERIOD > 0 and total_steps % cfg.TEST.EVAL_PERIOD == 0:
                logger.info('Start validation')
                result_dict = eval_disp(model, cfg, output_dir=args.checkpoint_dir)
                if comm.is_main_process():
                    wandb.log({f"val/{k}": v for k, v in result_dict.items()})
                    
                    try:
                        if 'disp' in result_dict and 'epe' in result_dict['disp']:
                            epe_score = result_dict['disp']['epe']
                        elif 'epe' in result_dict:
                            epe_score = result_dict['epe']
                        else:
                            epe_score = None
                            
                        if epe_score is not None and epe_score < best_epe:
                            best_epe = epe_score
                            logger.info(f"New best validation EPE: {best_epe:.4f}. Saving best checkpoint.")
                            torch.save({
                                'model': model_without_ddp.state_dict(),
                                'model_config': export_model_config(cfg),
                                'step': total_steps,
                                'epoch': epoch,
                                'epe': best_epe
                            }, best_checkpoint_path)
                            has_new_best = True
                    except Exception as e:
                        logger.warning(f"Error checking/saving best validation checkpoint: {e}")

                model.train()

            if total_steps >= cfg.SOLVER.MAX_ITER:
                logger.info('Training done')

                return
        
        epoch += 1
    finally:
        if comm.is_main_process():
            if os.path.exists(best_checkpoint_path):
                logger.info("Shutdown / finish detected. Uploading best checkpoint to cloud...")
                upload_checkpoint_to_r2(best_checkpoint_path, cloud_best_name)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )