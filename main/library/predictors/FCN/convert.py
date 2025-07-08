import torch
import scipy.stats

CENTS_PER_BIN, PITCH_BINS = 5, 1440

def bins_to_frequency(bins):
    if str(bins.device).startswith("ocl"): bins = bins.to(torch.float32)

    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    return 10 * 2 ** ((cents + cents.new_tensor(scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size()))) / 1200)

def frequency_to_bins(frequency, quantize_fn=torch.floor):
    return quantize_fn(((1200 * torch.log2(frequency / 10)) - 1997.3794084376191) / CENTS_PER_BIN).int()

def seconds_to_samples(seconds, sample_rate):
    return seconds * sample_rate