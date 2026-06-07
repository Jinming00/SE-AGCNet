"""Training script for SE-AGCNet."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
sys.path.append("..")

import time
import argparse
import json
import random
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from datetime import datetime
import setproctitle

# Local imports
from env import AttrDict, build_env
from dataset import DatasetWithOrigin, Dataset, get_dataset_filelist, mag_pha_stft, mag_pha_istft
from models.agc import MPSENetAGC
from models.discriminator import MetricDiscriminator, batch_pesq
from losses import compute_generator_loss, compute_discriminator_loss, compute_agc_loss
from validator import validate_using_inference
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

setproctitle.setproctitle('se-agcnet')


def tensor_to_float(value):
    """Convert tensors and numeric values to plain floats for external loggers."""
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


def init_validation_dnsmos_scorer(a, device):
    if not bool(int(getattr(a, 'validation_dnsmos_enabled', 0))):
        return None
    if device.type != 'cuda':
        raise RuntimeError('DNSMOS validation was enabled, but the validation device is not CUDA.')

    from dnsmos_utils import DNSMOSOVRLScorer

    scorer = DNSMOSOVRLScorer(
        getattr(a, 'validation_dnsmos_path', ''),
        prefer_cuda=True,
        allow_cpu_fallback=False,
    )
    print(
        f"[validation] DNSMOS OVRL enabled provider={scorer.provider} "
        f"path={scorer.model_dir} "
        f"ort_site={scorer.ort_imported_from or 'current_env'}"
    )
    return scorer


def split_train_validation_indexes(indexes, validation_ratio, seed=1234):
    indexes = list(indexes)
    ratio = float(validation_ratio)
    if ratio <= 0:
        return indexes, []
    if ratio >= 1:
        raise ValueError('--validation_ratio must be smaller than 1.0')

    rng = random.Random(seed)
    rng.shuffle(indexes)
    validation_count = int(len(indexes) * ratio)
    if validation_count == 0 and len(indexes) > 0:
        validation_count = 1
    validation_indexes = indexes[:validation_count]
    train_indexes = indexes[validation_count:]
    return train_indexes, validation_indexes


def get_extra_validation_indexes(clean_dir, noisy_dir):
    if not clean_dir or not noisy_dir:
        return []
    if not os.path.isdir(clean_dir) or not os.path.isdir(noisy_dir):
        return []
    clean_files = [f[:-4] for f in os.listdir(clean_dir) if f.endswith('.wav')]
    noisy_files = [f[:-4] for f in os.listdir(noisy_dir) if f.endswith('.wav')]
    return sorted(list(set(clean_files) & set(noisy_files)))


def log_validation_metrics(metrics, prefix, steps, sw, wandb_run):
    if metrics is None:
        return None

    val_pesq = metrics.get('pesq')
    if val_pesq is None or val_pesq <= 0:
        return None

    sw.add_scalar(f"{prefix}/pesq", val_pesq, steps)
    sw.add_scalar(f"{prefix}/pesq_count", metrics.get('pesq_count', 0), steps)
    sw.add_scalar(f"{prefix}/size", metrics.get('validation_size', 0), steps)
    sw.add_scalar(f"{prefix}/inference_count", metrics.get('inference_count', 0), steps)

    val_dnsmos = metrics.get('dnsmos_ovrl')
    if val_dnsmos is not None:
        sw.add_scalar(f"{prefix}/dnsmos_ovrl", val_dnsmos, steps)
        sw.add_scalar(f"{prefix}/dnsmos_count", metrics.get('dnsmos_count', 0), steps)
        sw.add_scalar(f"{prefix}/dnsmos_skipped", metrics.get('dnsmos_skipped', 0), steps)

    if wandb_run is not None:
        log_data = {
            f"{prefix}/pesq": float(val_pesq),
            f"{prefix}/pesq_count": int(metrics.get('pesq_count', 0)),
            f"{prefix}/size": int(metrics.get('validation_size', 0)),
            f"{prefix}/inference_count": int(metrics.get('inference_count', 0)),
        }
        if val_dnsmos is not None:
            log_data.update({
                f"{prefix}/dnsmos_ovrl": float(val_dnsmos),
                f"{prefix}/dnsmos_count": int(metrics.get('dnsmos_count', 0)),
                f"{prefix}/dnsmos_skipped": int(metrics.get('dnsmos_skipped', 0)),
            })
        wandb_run.log(log_data, step=steps)

    return val_pesq


def train(rank, a, h):
    """Main training function for distributed/single GPU."""
    if h.num_gpus > 1:
        from datetime import timedelta
        timeout = timedelta(minutes=30)
        init_process_group(
            backend=h.dist_config['dist_backend'], 
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus, 
            rank=rank, 
            timeout=timeout
        )
    
    torch.cuda.manual_seed(h.seed)
    device = torch.device(f'cuda:{rank}')
    
    # Initialize models
    generator = MPSENetAGC(h).to(device)
    discriminator = MetricDiscriminator().to(device)
    
    if rank == 0:
        print(generator)
        # Create checkpoint directories
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("Checkpoints directory:", a.checkpoint_path)
        
        # Create logging directories
        log_dir = os.path.join(a.checkpoint_path, 'training_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log_path = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        pesq_log_path = os.path.join(log_dir, f'validation_pesq_{timestamp}.txt')
        
        # Initialize log files
        with open(train_log_path, 'w') as f:
            f.write(f"Training Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Asymmetric Loss: {'Enabled' if a.enable_asymmetric_loss else 'Disabled'}")
            if a.enable_asymmetric_loss:
                f.write(f" (α={a.asym_alpha})")
            f.write("\n")
            f.write(f"AGC Penalty: {'Enabled' if a.enable_agc_penalty else 'Disabled'}")
            if a.enable_agc_penalty:
                f.write(
                    f" (penalty_factor={a.agc_penalty_factor}, "
                    f"silence_threshold={a.agc_silence_threshold})"
                )
            f.write("\n")
            f.write("="*80 + "\n")
            loss_type = "Asymmetric" if a.enable_asymmetric_loss else "Traditional"
            agc_type = "Penalty" if a.enable_agc_penalty else "L1"
            f.write(f"Format: Steps, Stage, Gen_Loss, Disc_Loss, Metric, Mag({loss_type}), Pha, Com({loss_type}), Time({loss_type}), STFT, [AGC({agc_type})], LR_Gen, LR_Disc, Time_per_batch\n")
            f.write("="*80 + "\n")
        
        with open(pesq_log_path, 'w') as f:
            f.write(f"Validation PESQ Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
            f.write("Format: Steps, PESQ_Score, Timestamp\n")
            f.write("="*80 + "\n")
    
    # Load checkpoints if available
    steps = 0
    last_epoch = -1
    state_dict_do = None
    
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        
        if cp_g is not None and cp_do is not None:
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            generator.load_state_dict(state_dict_g['generator'])
            discriminator.load_state_dict(state_dict_do['discriminator'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']
    
    # Wrap models for distributed training
    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator, device_ids=[rank], find_unused_parameters=True
        ).to(device)
        discriminator = DistributedDataParallel(
            discriminator, device_ids=[rank]
        ).to(device)
    
    generator_module = generator.module if h.num_gpus > 1 else generator

    # Initialize optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, 
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_agc = torch.optim.AdamW(
        generator_module.agc.parameters(), h.learning_rate,
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(), h.learning_rate, 
        betas=[h.adam_b1, h.adam_b2]
    )
    
    # Try to load optimizer states
    optimizer_loaded = False
    agc_optimizer_loaded = False
    if state_dict_do is not None:
        try:
            optim_g.load_state_dict(state_dict_do['optim_g'])
            optim_d.load_state_dict(state_dict_do['optim_d'])
            if 'optim_agc' in state_dict_do:
                optim_agc.load_state_dict(state_dict_do['optim_agc'])
                agc_optimizer_loaded = True
            optimizer_loaded = True
            if rank == 0:
                print("Successfully loaded optimizer states from checkpoint")
        except ValueError as e:
            if rank == 0:
                print(f"Warning: Could not load optimizer states ({e})")
                print("Continuing with fresh optimizers...")
        except Exception as e:
            if rank == 0:
                print(f"Warning: Unexpected error loading optimizer states: {e}")
                print("Continuing with fresh optimizers...")
    
    # Initialize learning rate schedulers
    # If optimizer states were not loaded, we need to set initial_lr manually
    # or reset last_epoch to avoid KeyError
    if not optimizer_loaded and last_epoch != -1:
        # Set initial_lr manually for each param group
        for group in optim_g.param_groups:
            group.setdefault('initial_lr', h.learning_rate)
        for group in optim_d.param_groups:
            group.setdefault('initial_lr', h.learning_rate)
        if rank == 0:
            print(f"Manually set initial_lr for schedulers (last_epoch={last_epoch})")
    if not agc_optimizer_loaded and last_epoch != -1:
        for group in optim_agc.param_groups:
            group.setdefault('initial_lr', h.learning_rate)
        if rank == 0 and optimizer_loaded:
            print("AGC optimizer state not found in checkpoint; initializing AGC scheduler from base learning rate")
    
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optim_g, step_size=30, gamma=0.5, last_epoch=last_epoch
    )
    scheduler_agc = torch.optim.lr_scheduler.StepLR(
        optim_agc, step_size=30, gamma=0.5, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.StepLR(
        optim_d, step_size=30, gamma=0.5, last_epoch=last_epoch
    )
    
    # Prepare datasets
    training_indexes = get_dataset_filelist(a)
    train_indexes, validation_indexes = split_train_validation_indexes(
        training_indexes,
        a.validation_ratio,
        h.seed,
    )
    if rank == 0:
        print(
            f"Train/validation split: train={len(train_indexes)}, "
            f"validation={len(validation_indexes)} (ratio={a.validation_ratio})"
        )
    
    trainset = DatasetWithOrigin(
        train_indexes,
        a.input_train_clean_dir, 
        a.input_train_noisy_dir, 
        a.input_train_origin_dir,
        h.segment_size, h.sampling_rate, 
        split=True, n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True, 
        device=None
    )
    
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    
    train_loader = DataLoader(
        trainset, 
        num_workers=h.num_workers, 
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True
    )
    
    # Prepare validation dataset
    if rank == 0:
        validset = None
        if len(validation_indexes) > 0:
            validset = Dataset(
                validation_indexes,
                a.input_train_origin_dir,
                a.input_train_noisy_dir,
                h.segment_size, h.sampling_rate,
                split=False, shuffle=False,
                n_cache_reuse=0, device=None
            )
        print(f"Validation from train split: {len(validation_indexes)} files")

        extra_validset = None
        extra_validation_indexes = []
        if bool(int(a.extra_validation_enabled)):
            extra_validation_indexes = get_extra_validation_indexes(
                a.input_test_clean_dir,
                a.input_test_noisy_dir,
            )
            if len(extra_validation_indexes) > 0:
                extra_validset = Dataset(
                    extra_validation_indexes,
                    a.input_test_clean_dir,
                    a.input_test_noisy_dir,
                    h.segment_size, h.sampling_rate,
                    split=False, shuffle=False,
                    n_cache_reuse=0, device=None
                )
        print(f"Extra validation from test dirs: {len(extra_validation_indexes)} files")
        
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        dnsmos_scorer = init_validation_dnsmos_scorer(a, device)
        wandb_run = None
        if a.use_wandb:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "wandb is not installed. Install it with `pip install wandb` "
                    "or run without --use_wandb."
                ) from exc

            config_dict = dict(json_config=dict(h), args=vars(a))
            wandb_run = wandb.init(
                project=a.wandb_project,
                entity=a.wandb_entity,
                name=a.wandb_run_name,
                group=a.group_name,
                dir=a.checkpoint_path,
                config=config_dict,
                mode=a.wandb_mode,
                resume="allow",
            )
            if a.wandb_watch != 'false':
                wandb.watch(
                    generator.module if h.num_gpus > 1 else generator,
                    log=a.wandb_watch,
                    log_freq=a.summary_interval,
                )
    
    # Training loop
    generator.train()
    discriminator.train()
    best_pesq = 0
    
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            asym_info = f" [Asym α={a.asym_alpha}]" if a.enable_asymmetric_loss else " [Traditional Loss]"
            agc_penalty_info = f" [AGC Penalty={a.agc_penalty_factor}]" if a.enable_agc_penalty else ""
            
            if a.staged_training and epoch < a.stage1_epochs:
                if a.stage1_train_standalone_agc:
                    print(f"Epoch: {epoch+1} - Stage 1 (MP-SENet + standalone AGC, {a.stage1_epochs} epochs total){asym_info}{agc_penalty_info}")
                else:
                    print(f"Epoch: {epoch+1} - Stage 1 (MP-SENet only, {a.stage1_epochs} epochs total){asym_info}{agc_penalty_info}")
            elif a.staged_training and epoch == a.stage1_epochs:
                print(f"Epoch: {epoch+1} - Stage 2 (MP-SENet + AGC, AGC weight: {a.agc_loss_weight}){asym_info}{agc_penalty_info}")
            else:
                print(f"Epoch: {epoch+1}{asym_info}{agc_penalty_info}")
        
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        
        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            
            # Unpack batch
            clean_audio, noisy_audio, origin_audio, norm_factor = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            origin_audio = origin_audio.to(device, non_blocking=True)
            norm_factor = norm_factor.to(device, non_blocking=True)
            
            zero_labels = torch.zeros(h.batch_size).to(device, non_blocking=True)
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)
            
            # Compute spectrograms
            clean_mag, clean_pha, clean_com = mag_pha_stft(
                clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            noisy_mag, noisy_pha, _ = mag_pha_stft(
                noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            origin_mag, _, _ = mag_pha_stft(
                origin_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            stage1_active = a.staged_training and epoch < a.stage1_epochs
            run_agc = not stage1_active
            run_stage1_standalone_agc = stage1_active and bool(a.stage1_train_standalone_agc)

            batch_size = origin_mag.shape[0]
            origin_rms = torch.sqrt(torch.mean(origin_mag.reshape(batch_size, -1) ** 2, dim=1))
            origin_norm_factor = 1.0 / (origin_rms + 1e-8)
            origin_mag_normalized = origin_mag * origin_norm_factor.view(-1, 1, 1)
            
            # Forward pass through generator
            (agc_mag_normalized, mpnet_pha, agc_com_normalized, 
             mpnet_mag, mpnet_com, agc_norm_factor) = generator(
                noisy_mag, noisy_pha, norm_factor, run_agc=run_agc
            )
            
            # Reconstruct audio
            mpnet_audio = mag_pha_istft(
                mpnet_mag, mpnet_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            
            mpnet_mag_hat, _, mpnet_com_hat = mag_pha_stft(
                mpnet_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            
            # Calculate PESQ scores for discriminator
            audio_list_r = list(clean_audio.cpu().numpy())
            audio_list_g = list(mpnet_audio.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)
            
            # Train discriminator
            optim_d.zero_grad()
            loss_disc_all = compute_discriminator_loss(
                discriminator, clean_mag, mpnet_mag_hat, 
                one_labels, batch_pesq_score, device
            )
            loss_disc_all.backward()
            optim_d.step()
            
            # Train generator
            optim_g.zero_grad()
            optim_agc.zero_grad()
            
            # Get metric from discriminator for generator training
            mpnet_mag_hat_new, _, mpnet_com_hat_new = mag_pha_stft(
                mpnet_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            metric_g = discriminator(clean_mag, mpnet_mag_hat_new)
            
            # Compute generator loss
            loss_gen_all, loss_dict = compute_generator_loss(
                clean_mag, clean_pha, clean_com, clean_audio,
                mpnet_mag, mpnet_pha, mpnet_com, mpnet_audio,
                mpnet_com_hat, agc_mag_normalized, origin_mag_normalized if run_agc else None,
                metric_g, one_labels, h, a, epoch
            )
            
            loss_gen_all.backward()
            optim_g.step()

            standalone_agc_loss = None
            if run_stage1_standalone_agc:
                standalone_agc_mag, _ = generator_module.run_agc_from_mag(clean_mag)
                standalone_agc_loss = compute_agc_loss(standalone_agc_mag, origin_mag_normalized, a)
                standalone_agc_loss.backward()
                optim_agc.step()
            
            # Logging
            if rank == 0:
                if steps % a.stdout_interval == 0:
                    batch_time = time.time() - start_b
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    loss_type_suffix = " (Asym)" if a.enable_asymmetric_loss else " (Trad)"
                    stage_info = loss_dict['stage_info']
                    
                    if not run_agc:
                        log_msg = (f'Steps : {steps}, {stage_info} - Gen Loss: {loss_gen_all:.3f}, '
                                  f'Disc Loss: {loss_disc_all:.3f}, Metric: {loss_dict["loss_metric"]:.3f}, '
                                  f'Mag{loss_type_suffix}: {loss_dict["loss_mag"]:.3f}, '
                                  f'Pha: {loss_dict["loss_pha"]:.3f}, '
                                  f'Com{loss_type_suffix}: {loss_dict["loss_com"]:.3f}, '
                                  f'Time{loss_type_suffix}: {loss_dict["loss_time"]:.3f}, '
                                  f'STFT: {loss_dict["loss_stft"]:.3f}')
                        if standalone_agc_loss is not None:
                            log_msg += f', AGC-Standalone: {standalone_agc_loss:.3f}'
                        log_msg += f', s/b: {batch_time:.3f}'
                    else:
                        log_msg = (f'Steps : {steps}, {stage_info} - Gen Loss: {loss_gen_all:.3f}, '
                                  f'Disc Loss: {loss_disc_all:.3f}, Metric: {loss_dict["loss_metric"]:.3f}, '
                                  f'Mag{loss_type_suffix}: {loss_dict["loss_mag"]:.3f}, '
                                  f'Pha: {loss_dict["loss_pha"]:.3f}, '
                                  f'Com{loss_type_suffix}: {loss_dict["loss_com"]:.3f}, '
                                  f'Time{loss_type_suffix}: {loss_dict["loss_time"]:.3f}, '
                                  f'STFT: {loss_dict["loss_stft"]:.3f}, '
                                  f'AGC: {loss_dict["loss_agc"]:.3f}, s/b: {batch_time:.3f}')
                    
                    print(log_msg)
                    
                    # Write to log file periodically
                    log_interval = a.stdout_interval * (10 if epoch < a.stage1_epochs else 100)
                    if steps % log_interval == 0:
                        current_lr_g = optim_g.param_groups[0]['lr']
                        current_lr_d = optim_d.param_groups[0]['lr']
                        with open(train_log_path, 'a') as f:
                            if run_agc:
                                agc_str = f'{loss_dict["loss_agc"]:.3f}'
                            elif standalone_agc_loss is not None:
                                agc_str = f'{tensor_to_float(standalone_agc_loss):.3f}'
                            else:
                                agc_str = 'N/A'
                            f.write(f"{current_time} | {steps:8d} | {stage_info:20s} | "
                                   f"{loss_gen_all:8.3f} | {loss_disc_all:8.3f} | "
                                   f"{loss_dict['loss_metric']:8.3f} | {loss_dict['loss_mag']:8.3f} | "
                                   f"{loss_dict['loss_pha']:8.3f} | {loss_dict['loss_com']:8.3f} | "
                                   f"{loss_dict['loss_time']:8.3f} | {loss_dict['loss_stft']:8.3f} | "
                                   f"{agc_str:8s} | {current_lr_g:.2e} | {current_lr_d:.2e} | "
                                   f"{batch_time:8.3f}\n")
                
                # Save checkpoints
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = f"{a.checkpoint_path}/g_{steps:08d}"
                    save_checkpoint(checkpoint_path, {
                        'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()
                    })
                    checkpoint_path = f"{a.checkpoint_path}/do_{steps:08d}"
                    save_checkpoint(checkpoint_path, {
                        'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                        'optim_g': optim_g.state_dict(),
                        'optim_agc': optim_agc.state_dict(),
                        'optim_d': optim_d.state_dict(),
                        'steps': steps,
                        'epoch': epoch
                    })
                
                # TensorBoard logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", loss_dict['loss_metric'], steps)
                    sw.add_scalar("Training/Magnitude Loss", loss_dict['loss_mag'], steps)
                    sw.add_scalar("Training/Phase Loss", loss_dict['loss_pha'], steps)
                    sw.add_scalar("Training/Complex Loss", loss_dict['loss_com'], steps)
                    sw.add_scalar("Training/Time Loss", loss_dict['loss_time'], steps)
                    sw.add_scalar("Training/Consistency Loss", loss_dict['loss_stft'], steps)
                    
                    if run_agc:
                        sw.add_scalar("Training/AGC Loss", loss_dict['loss_agc'], steps)
                    elif standalone_agc_loss is not None:
                        sw.add_scalar("Training/AGC Standalone Loss", standalone_agc_loss, steps)
                    
                    sw.add_scalar("Training/Stage", 
                                 1 if stage1_active else 2,
                                 steps)

                    if wandb_run is not None:
                        wandb_log = {
                            "train/generator_loss": tensor_to_float(loss_gen_all),
                            "train/discriminator_loss": tensor_to_float(loss_disc_all),
                            "train/metric_loss": tensor_to_float(loss_dict['loss_metric']),
                            "train/magnitude_loss": tensor_to_float(loss_dict['loss_mag']),
                            "train/phase_loss": tensor_to_float(loss_dict['loss_pha']),
                            "train/complex_loss": tensor_to_float(loss_dict['loss_com']),
                            "train/time_loss": tensor_to_float(loss_dict['loss_time']),
                            "train/consistency_loss": tensor_to_float(loss_dict['loss_stft']),
                            "train/stage": 1 if stage1_active else 2,
                            "train/epoch": epoch + 1,
                            "train/lr_generator": optim_g.param_groups[0]['lr'],
                            "train/lr_agc": optim_agc.param_groups[0]['lr'],
                            "train/lr_discriminator": optim_d.param_groups[0]['lr'],
                        }
                        if run_agc:
                            wandb_log["train/agc_loss"] = tensor_to_float(loss_dict['loss_agc'])
                        elif standalone_agc_loss is not None:
                            wandb_log["train/agc_standalone_loss"] = tensor_to_float(standalone_agc_loss)
                        wandb_run.log(wandb_log, step=steps)
                
                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    if stage1_active:
                        print("Validation skipped during Stage 1")
                        steps += 1
                        continue

                    generator.eval()
                    torch.cuda.empty_cache()

                    val_pesq = None
                    if validset is not None and len(validset) > 0:
                        print("Starting validation...")
                        validation_metrics = validate_using_inference(
                            generator,
                            h,
                            validset,
                            device,
                            dnsmos_scorer=dnsmos_scorer,
                            return_metrics=True,
                            label='validation',
                        )
                        val_pesq = log_validation_metrics(validation_metrics, 'validation', steps, sw, wandb_run)
                    else:
                        print("Validation skipped: validation_ratio is 0 or no validation files were selected")

                    extra_metrics = None
                    if extra_validset is not None and len(extra_validset) > 0:
                        print("Starting extra validation...")
                        extra_metrics = validate_using_inference(
                            generator,
                            h,
                            extra_validset,
                            device,
                            dnsmos_scorer=dnsmos_scorer,
                            return_metrics=True,
                            label='extra_validation',
                        )
                        log_validation_metrics(extra_metrics, 'extra_validation', steps, sw, wandb_run)
                    else:
                        print("Extra validation skipped: disabled or no extra validation files found")

                    if val_pesq is not None and val_pesq > 0:
                        val_dnsmos = validation_metrics.get('dnsmos_ovrl')
                        pesq_msg = f'Steps : {steps}, Validation PESQ : {val_pesq:.3f}'
                        if val_dnsmos is not None:
                            pesq_msg += f', DNSMOS OVRL : {float(val_dnsmos):.3f}'
                        if extra_metrics is not None:
                            extra_pesq = extra_metrics.get('pesq')
                            extra_dnsmos = extra_metrics.get('dnsmos_ovrl')
                            if extra_pesq is not None:
                                pesq_msg += f', Extra PESQ : {float(extra_pesq):.3f}'
                            if extra_dnsmos is not None:
                                pesq_msg += f', Extra DNSMOS OVRL : {float(extra_dnsmos):.3f}'
                        print(pesq_msg)
                        
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(pesq_log_path, 'a') as f:
                            dnsmos_str = 'N/A' if val_dnsmos is None else f'{float(val_dnsmos):8.4f}'
                            extra_pesq = extra_metrics.get('pesq') if extra_metrics is not None else None
                            extra_dnsmos = extra_metrics.get('dnsmos_ovrl') if extra_metrics is not None else None
                            extra_pesq_str = 'N/A' if extra_pesq is None else f'{float(extra_pesq):8.4f}'
                            extra_dnsmos_str = 'N/A' if extra_dnsmos is None else f'{float(extra_dnsmos):8.4f}'
                            f.write(
                                f"{current_time} | {steps:8d} | {val_pesq:8.4f} | "
                                f"DNSMOS {dnsmos_str} | Extra PESQ {extra_pesq_str} | "
                                f"Extra DNSMOS {extra_dnsmos_str} | Using inference.py\n"
                            )
                        
                        # Save best checkpoint
                        if val_pesq > best_pesq and epoch >= a.best_checkpoint_start_epoch:
                            best_pesq = val_pesq
                            checkpoint_path = f"{a.checkpoint_path}/best_g"
                            save_checkpoint(checkpoint_path, {
                                'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()
                            })
                    elif extra_metrics is not None and extra_metrics.get('pesq') is not None:
                        extra_pesq = extra_metrics.get('pesq')
                        extra_dnsmos = extra_metrics.get('dnsmos_ovrl')
                        pesq_msg = f'Steps : {steps}, Extra PESQ : {float(extra_pesq):.3f}'
                        if extra_dnsmos is not None:
                            pesq_msg += f', Extra DNSMOS OVRL : {float(extra_dnsmos):.3f}'
                        print(pesq_msg)

                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(pesq_log_path, 'a') as f:
                            extra_dnsmos_str = 'N/A' if extra_dnsmos is None else f'{float(extra_dnsmos):8.4f}'
                            f.write(
                                f"{current_time} | {steps:8d} | Validation N/A | "
                                f"DNSMOS N/A | Extra PESQ {float(extra_pesq):8.4f} | "
                                f"Extra DNSMOS {extra_dnsmos_str} | Using inference.py\n"
                            )
                    else:
                        print("No valid primary validation PESQ scores calculated")
                    
                    generator.train()
            
            steps += 1
        
        # Step schedulers
        scheduler_g.step()
        scheduler_agc.step()
        scheduler_d.step()
        
        if rank == 0:
            print(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')

    if rank == 0:
        sw.close()
        if wandb_run is not None:
            wandb_run.finish()


def main():
    """Main entry point."""
    print('Initializing SE-AGCNet training...')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_train_clean_dir', default='/home/ccds-jmzhang/10samples/clean')
    parser.add_argument('--input_train_noisy_dir', default='/home/ccds-jmzhang/10samples/noisy')
    parser.add_argument('--input_train_origin_dir', default='/home/ccds-jmzhang/10samples/noisy')
    parser.add_argument('--input_test_clean_dir', default='/home/ccds-jmzhang/10samples/clean')
    parser.add_argument('--input_test_noisy_dir', default='/home/ccds-jmzhang/10samples/noisy')
    parser.add_argument('--checkpoint_path', default='/home/ccds-jmzhang/test')
    parser.add_argument('--config', default='/home/ccds-jmzhang/SE-AGCNet/SE_AGCNet/config.json')
    
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=10, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=10, type=int)
    parser.add_argument('--validation_ratio', default=0.0, type=float,
                       help='Ratio of training files held out for primary validation. Set 0 to disable primary validation')
    parser.add_argument('--extra_validation_enabled', default=1, type=int, choices=[0, 1],
                       help='Set to 1 to run extra validation on input_test_clean_dir/input_test_noisy_dir')
    
    parser.add_argument('--staged_training', default=True, type=bool, 
                       help='Enable staged training')
    parser.add_argument('--stage1_epochs', default=5, type=int, 
                       help='Number of epochs for stage 1 training')
    parser.add_argument('--stage1_train_standalone_agc', default=0, type=int, choices=[0, 1],
                       help='Set to 1 to also train AGC independently on lower->origin during stage 1')
    parser.add_argument('--agc_loss_weight', default=0.9, type=float, 
                       help='Weight for AGC loss in stage 2')
    
    parser.add_argument('--asym_alpha', default=10.0, type=float, 
                       help='Asymmetric penalty factor for over-suppression')
    parser.add_argument('--enable_asymmetric_loss', default=True, type=bool, 
                       help='Enable asymmetric loss function')
    
    parser.add_argument('--agc_penalty_factor', default=10.0, type=float, 
                       help='Penalty factor for AGC positive prediction when target is zero')
    parser.add_argument('--agc_silence_threshold', default=1e-4, type=float,
                       help='Threshold below which AGC targets are treated as silence for penalty')
    parser.add_argument('--enable_agc_penalty', default=True, type=bool, 
                       help='Enable AGC conditional positive penalty loss')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases experiment tracking')
    parser.add_argument('--wandb_project', default='SE-AGCNet',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', default=None,
                       help='Weights & Biases entity/team name')
    parser.add_argument('--wandb_run_name', default=None,
                       help='Weights & Biases run name')
    parser.add_argument('--wandb_mode', default='online', choices=['online', 'offline', 'disabled'],
                       help='Weights & Biases mode')
    parser.add_argument('--wandb_watch', default='gradients', choices=['gradients', 'parameters', 'all', 'false'],
                       help='Model data to watch in Weights & Biases')
    parser.add_argument('--validation_dnsmos_enabled', default=1, type=int, choices=[0, 1],
                       help='Set to 1 to compute DNSMOS OVRL on enhanced validation audio using GPU ONNX Runtime')
    parser.add_argument('--validation_dnsmos_path',
                       default='/home/ccds-jmzhang/MP-SENet/dnsmos/DNSMOS',
                       help='Directory containing DNSMOS sig_bak_ovr.onnx')
    
    a = parser.parse_args()
    
    # Load config
    with open(a.config) as f:
        data = f.read()
    
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)
    
    # Set random seeds
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    
    # Start training
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
