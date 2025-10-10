"""Training script for SE-AGCNet."""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
sys.path.append("..")

import time
import argparse
import json
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
from losses import compute_generator_loss, compute_discriminator_loss
from validator import validate_using_inference
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

setproctitle.setproctitle('se-agcnet')


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
                f.write(f" (penalty_factor={a.agc_penalty_factor})")
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
    
    # Initialize optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, 
        betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(), h.learning_rate, 
        betas=[h.adam_b1, h.adam_b2]
    )
    
    # Try to load optimizer states
    optimizer_loaded = False
    if state_dict_do is not None:
        try:
            optim_g.load_state_dict(state_dict_do['optim_g'])
            optim_d.load_state_dict(state_dict_do['optim_d'])
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
    
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optim_g, step_size=30, gamma=0.5, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.StepLR(
        optim_d, step_size=30, gamma=0.5, last_epoch=last_epoch
    )
    
    # Prepare datasets
    training_indexes = get_dataset_filelist(a)
    
    trainset = DatasetWithOrigin(
        training_indexes, 
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
        # Get validation file indexes from test directories
        test_clean_files = [
            f[:-4] for f in os.listdir(a.input_test_clean_dir) 
            if f.endswith('.wav')
        ]
        test_noisy_files = [
            f[:-4] for f in os.listdir(a.input_test_noisy_dir) 
            if f.endswith('.wav')
        ]
        validation_indexes = list(set(test_clean_files) & set(test_noisy_files))
        print(f"Validation: {len(validation_indexes)} files")
        
        validset = Dataset(
            validation_indexes, 
            a.input_test_clean_dir, 
            a.input_test_noisy_dir,
            h.segment_size, h.sampling_rate, 
            split=False, shuffle=False, 
            n_cache_reuse=0, device=None
        )
        
        validation_loader = DataLoader(
            validset, 
            num_workers=16, 
            shuffle=False,
            sampler=None,
            batch_size=16,
            pin_memory=True,
            drop_last=False
        )
        
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
    
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
            
            # Forward pass through generator
            (agc_mag_normalized, mpnet_pha, agc_com_normalized, 
             mpnet_mag, mpnet_com, agc_norm_factor) = generator(
                noisy_mag, noisy_pha, norm_factor
            )
            
            # Reconstruct audio
            final_mag = agc_mag_normalized / agc_norm_factor.view(-1, 1, 1)
            final_audio = mag_pha_istft(
                final_mag, mpnet_pha, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
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
            
            # Compute origin_mag_normalized for AGC loss
            batch_size, freq_bins, time_frames = origin_mag.shape
            origin_rms = torch.sqrt(torch.mean(origin_mag.view(batch_size, -1) ** 2, dim=1))
            origin_norm_factor = 1.0 / (origin_rms + 1e-8)
            origin_mag_normalized = origin_mag * origin_norm_factor.view(-1, 1, 1)
            
            # Get metric from discriminator for generator training
            mpnet_mag_hat_new, _, mpnet_com_hat_new = mag_pha_stft(
                mpnet_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor
            )
            metric_g = discriminator(clean_mag, mpnet_mag_hat_new)
            
            # Compute generator loss
            loss_gen_all, loss_dict = compute_generator_loss(
                clean_mag, clean_pha, clean_com, clean_audio,
                mpnet_mag, mpnet_pha, mpnet_com, mpnet_audio,
                mpnet_com_hat, agc_mag_normalized, origin_mag_normalized,
                metric_g, one_labels, h, a, epoch
            )
            
            loss_gen_all.backward()
            optim_g.step()
            
            # Logging
            if rank == 0:
                if steps % a.stdout_interval == 0:
                    batch_time = time.time() - start_b
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    loss_type_suffix = " (Asym)" if a.enable_asymmetric_loss else " (Trad)"
                    stage_info = loss_dict['stage_info']
                    
                    if a.staged_training and epoch < a.stage1_epochs:
                        log_msg = (f'Steps : {steps}, {stage_info} - Gen Loss: {loss_gen_all:.3f}, '
                                  f'Disc Loss: {loss_disc_all:.3f}, Metric: {loss_dict["loss_metric"]:.3f}, '
                                  f'Mag{loss_type_suffix}: {loss_dict["loss_mag"]:.3f}, '
                                  f'Pha: {loss_dict["loss_pha"]:.3f}, '
                                  f'Com{loss_type_suffix}: {loss_dict["loss_com"]:.3f}, '
                                  f'Time{loss_type_suffix}: {loss_dict["loss_time"]:.3f}, '
                                  f'STFT: {loss_dict["loss_stft"]:.3f}, s/b: {batch_time:.3f}')
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
                            agc_str = 'N/A' if epoch < a.stage1_epochs else f'{loss_dict["loss_agc"]:.3f}'
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
                    
                    if not (a.staged_training and epoch < a.stage1_epochs):
                        sw.add_scalar("Training/AGC Loss", loss_dict['loss_agc'], steps)
                    
                    sw.add_scalar("Training/Stage", 
                                 1 if (a.staged_training and epoch < a.stage1_epochs) else 2, 
                                 steps)
                
                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    
                    print("Starting validation...")
                    val_pesq = validate_using_inference(generator, h, validset, device)
                    
                    if val_pesq is not None and val_pesq > 0:
                        sw.add_scalar("validation/pesq", val_pesq, steps)
                        print(f'Steps : {steps}, PESQ (complete wav files, wideband mode) : {val_pesq:.3f}')
                        
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        with open(pesq_log_path, 'a') as f:
                            f.write(f"{current_time} | {steps:8d} | {val_pesq:8.4f} | Using inference.py\n")
                        
                        # Save best checkpoint
                        if val_pesq > best_pesq and epoch >= a.best_checkpoint_start_epoch:
                            best_pesq = val_pesq
                            checkpoint_path = f"{a.checkpoint_path}/best_g"
                            save_checkpoint(checkpoint_path, {
                                'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()
                            })
                    else:
                        print("No valid PESQ scores calculated")
                    
                    generator.train()
            
            steps += 1
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print(f'Time taken for epoch {epoch + 1} is {int(time.time() - start)} sec\n')


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
    
    parser.add_argument('--staged_training', default=True, type=bool, 
                       help='Enable staged training')
    parser.add_argument('--stage1_epochs', default=5, type=int, 
                       help='Number of epochs for stage 1 (MP-SENet only)')
    parser.add_argument('--agc_loss_weight', default=0.9, type=float, 
                       help='Weight for AGC loss in stage 2')
    
    parser.add_argument('--asym_alpha', default=10.0, type=float, 
                       help='Asymmetric penalty factor for over-suppression')
    parser.add_argument('--enable_asymmetric_loss', default=True, type=bool, 
                       help='Enable asymmetric loss function')
    
    parser.add_argument('--agc_penalty_factor', default=10.0, type=float, 
                       help='Penalty factor for AGC positive prediction when target is zero')
    parser.add_argument('--enable_agc_penalty', default=True, type=bool, 
                       help='Enable AGC conditional positive penalty loss')
    
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
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    
    # Start training
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()

