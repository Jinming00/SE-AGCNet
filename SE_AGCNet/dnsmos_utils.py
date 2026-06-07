import os

import librosa
import numpy as np

try:
    import torch
except Exception:
    torch = None


try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - import error is surfaced at runtime
    ort = None
    _ORT_IMPORT_ERROR = exc
    _ORT_IMPORTED_FROM = None
else:
    _ORT_IMPORT_ERROR = None
    _ORT_IMPORTED_FROM = None


DNSMOS_SAMPLE_RATE = 16000
DNSMOS_INPUT_LENGTH_SECONDS = 9.01


def _ensure_mono_float32(audio):
    audio = np.asarray(audio)
    if audio.ndim == 0:
        raise ValueError("Audio must have at least 1 dimension")
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > 8:
            audio = np.mean(audio, axis=0)
        else:
            audio = np.mean(audio, axis=-1)
        return np.asarray(audio, dtype=np.float32)
    audio = np.reshape(audio, (-1,))
    return np.asarray(audio, dtype=np.float32)


class DNSMOSOVRLScorer:
    def __init__(self, model_dir, prefer_cuda=True, allow_cpu_fallback=None):
        if ort is None:
            raise ImportError(f"onnxruntime import failed: {_ORT_IMPORT_ERROR}")
        if allow_cpu_fallback is None:
            allow_cpu_fallback = not bool(prefer_cuda)

        self.model_dir = os.path.abspath(str(model_dir))
        self.ort_imported_from = _ORT_IMPORTED_FROM
        self.primary_model_path = os.path.join(self.model_dir, "sig_bak_ovr.onnx")
        if not os.path.isfile(self.primary_model_path):
            raise FileNotFoundError(f"DNSMOS primary model not found: {self.primary_model_path}")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        available_providers = list(ort.get_available_providers())
        providers = []
        cuda_ready = bool(prefer_cuda) and "CUDAExecutionProvider" in available_providers
        if cuda_ready and torch is not None:
            try:
                cuda_ready = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0
                if cuda_ready:
                    _ = int(torch.cuda.current_device())
            except Exception:
                cuda_ready = False
        if cuda_ready:
            providers.append((
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": "DEFAULT",
                    "cudnn_conv_use_max_workspace": "0",
                },
            ))
        elif bool(prefer_cuda) and not bool(allow_cpu_fallback):
            raise RuntimeError(
                "DNSMOS requires CUDAExecutionProvider, but this process has no usable CUDA device. "
                "Check the job allocation, CUDA_VISIBLE_DEVICES, and torch.cuda availability."
            )
        if bool(allow_cpu_fallback) and "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")
        if len(providers) == 0:
            providers = None

        session_kwargs = {"sess_options": sess_options}
        if providers is not None:
            session_kwargs["providers"] = providers

        self.session = ort.InferenceSession(self.primary_model_path, **session_kwargs)
        actual_providers = list(self.session.get_providers())
        self.provider = actual_providers[0] if len(actual_providers) > 0 else "unknown"
        if bool(prefer_cuda) and not bool(allow_cpu_fallback) and self.provider != "CUDAExecutionProvider":
            raise RuntimeError(
                f"DNSMOS requires CUDAExecutionProvider, but onnxruntime initialized {self.provider} instead."
            )

    @staticmethod
    def _polyfit_ovrl(ovrl_raw):
        poly = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        return float(poly(float(ovrl_raw)))

    def _prepare_audio(self, audio, sampling_rate):
        audio = _ensure_mono_float32(audio)
        if audio.size == 0:
            raise ValueError("Audio is empty")

        sampling_rate = int(sampling_rate)
        if sampling_rate != DNSMOS_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=DNSMOS_SAMPLE_RATE)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Audio is empty after resampling")

        required_samples = int(round(DNSMOS_INPUT_LENGTH_SECONDS * DNSMOS_SAMPLE_RATE))
        while audio.shape[0] < required_samples:
            audio = np.concatenate([audio, audio], axis=0)
        return audio

    def score_waveform(self, audio, sampling_rate):
        audio = self._prepare_audio(audio=audio, sampling_rate=sampling_rate)

        hop_len_samples = DNSMOS_SAMPLE_RATE
        required_samples = int(round(DNSMOS_INPUT_LENGTH_SECONDS * DNSMOS_SAMPLE_RATE))
        num_hops = int(np.floor(float(audio.shape[0]) / float(DNSMOS_SAMPLE_RATE)) - DNSMOS_INPUT_LENGTH_SECONDS) + 1
        num_hops = max(1, num_hops)

        predicted_ovrl_raw = []
        predicted_ovrl = []
        for idx in range(num_hops):
            start = int(idx * hop_len_samples)
            end = start + required_samples
            audio_seg = audio[start:end]
            if audio_seg.shape[0] < required_samples:
                continue

            input_features = np.asarray(audio_seg, dtype=np.float32)[np.newaxis, :]
            oi = {"input_1": input_features}
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.session.run(None, oi)[0][0]
            _ = mos_sig_raw, mos_bak_raw
            predicted_ovrl_raw.append(float(mos_ovr_raw))
            predicted_ovrl.append(self._polyfit_ovrl(mos_ovr_raw))

        if len(predicted_ovrl) == 0:
            raise RuntimeError("DNSMOS produced no valid segments")

        return {
            "OVRL_raw": float(np.mean(predicted_ovrl_raw)),
            "OVRL": float(np.mean(predicted_ovrl)),
            "num_hops": int(len(predicted_ovrl)),
            "provider": str(self.provider),
        }
