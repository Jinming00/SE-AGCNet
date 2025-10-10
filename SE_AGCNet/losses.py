"""Loss functions for SE-AGCNet."""

import torch
import torch.nn.functional as F


def g_asym(x, alpha=10.0):
    """Asymmetric penalty: g(x) = x if x<0, else alpha*x."""
    return torch.where(x > 0, alpha * x, x)


def asymmetric_magnitude_loss(pred_mag, target_mag, alpha=10.0):
    """Magnitude loss with asymmetric penalty for over-suppression."""
    x = target_mag - pred_mag
    asymmetric_penalty = g_asym(x, alpha)
    return torch.mean(torch.abs(asymmetric_penalty))


def asymmetric_complex_loss(pred_com, target_com, alpha=10.0):
    """Complex loss with asymmetric penalty for over-suppression."""
    x = target_com - pred_com
    asymmetric_penalty = g_asym(x, alpha)
    return torch.mean(torch.abs(asymmetric_penalty))


def asymmetric_time_loss(pred_audio, target_audio, alpha=10.0):
    """Time-domain loss with asymmetric penalty for over-suppression."""
    x = target_audio - pred_audio
    asymmetric_penalty = g_asym(x, alpha)
    return torch.mean(torch.abs(asymmetric_penalty))


def conditional_positive_penalty_loss(pred, target, penalty_factor=10.0):
    """
    AGC loss with penalty for positive predictions when target is zero.
    Prevents generating signal in silent regions.
    """
    normal_loss = F.l1_loss(pred, target, reduction='none')
    zero_target_mask = (target == 0).float()
    positive_pred_mask = (pred > 0).float()
    penalty_mask = zero_target_mask * positive_pred_mask
    weight = penalty_mask * penalty_factor + (1 - penalty_mask) * 1.0
    return torch.mean(normal_loss * weight)


def compute_generator_loss(clean_mag, clean_pha, clean_com, clean_audio,
                           mpnet_mag, mpnet_pha, mpnet_com, mpnet_audio,
                           mpnet_com_hat, agc_mag_normalized, origin_mag_normalized,
                           metric_g, one_labels, h, a, epoch):
    """Compute generator loss with all components."""
    from models.model import phase_losses
    
    # Select loss functions based on configuration
    if a.enable_asymmetric_loss:
        loss_mag = asymmetric_magnitude_loss(mpnet_mag, clean_mag, alpha=a.asym_alpha)
        loss_com = asymmetric_complex_loss(mpnet_com, clean_com, alpha=a.asym_alpha) * 2
        loss_time = asymmetric_time_loss(mpnet_audio, clean_audio, alpha=a.asym_alpha)
    else:
        loss_mag = F.mse_loss(clean_mag, mpnet_mag)
        loss_com = F.mse_loss(clean_com, mpnet_com) * 2
        loss_time = F.l1_loss(clean_audio, mpnet_audio)
    
    # Phase losses
    loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, mpnet_pha)
    loss_pha = loss_ip + loss_gd + loss_iaf
    
    # Consistency loss
    loss_stft = F.mse_loss(mpnet_com, mpnet_com_hat) * 2
    
    # Metric discriminator loss
    loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
    
    # AGC loss with optional conditional penalty
    if a.enable_agc_penalty:
        loss_agc = conditional_positive_penalty_loss(
            agc_mag_normalized, origin_mag_normalized, penalty_factor=a.agc_penalty_factor
        )
    else:
        loss_agc = F.l1_loss(agc_mag_normalized, origin_mag_normalized)
    
    # Staged training: MP-SENet only or MP-SENet + AGC
    if a.staged_training and epoch < a.stage1_epochs:
        loss_gen_all = (loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1 +
                       loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2)
        stage_info = "Stage 1 (MP-SENet)"
    else:
        loss_gen_all = (loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1 +
                       loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2 +
                       loss_agc * a.agc_loss_weight)
        stage_info = "Stage 2 (MP-SENet+AGC)"
    
    loss_dict = {
        'loss_mag': loss_mag.item(),
        'loss_pha': loss_pha.item(),
        'loss_com': loss_com.item(),
        'loss_time': loss_time.item(),
        'loss_stft': loss_stft.item(),
        'loss_metric': loss_metric.item(),
        'loss_agc': loss_agc.item(),
        'stage_info': stage_info
    }
    
    return loss_gen_all, loss_dict


def compute_discriminator_loss(discriminator, clean_mag, mpnet_mag_hat, 
                               one_labels, batch_pesq_score, device):
    """Compute metric discriminator loss."""
    metric_r = discriminator(clean_mag, clean_mag)
    metric_g = discriminator(clean_mag, mpnet_mag_hat.detach())
    
    loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
    
    if batch_pesq_score is not None:
        loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
    else:
        print('Warning: PESQ is None!')
        loss_disc_g = 0
    
    loss_disc_all = loss_disc_r + loss_disc_g
    return loss_disc_all

