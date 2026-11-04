"""Minimal toy ``(T, C, F)`` time-series generator.

Feature layout in the returned data array:

- 0: clean_signal
- 1: noise
- 2: noisy_signal
- 3: SNR db or SNR
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class DatasetConfig:
    n_timesteps: int = 1000
    n_channels: int = 8
    dt: float = 0.01
    seed: int | None = 123
    output_path: str | None = "toy_timeseries.npz"


@dataclass
class SignalConfig:
    amplitude: float = 1.0
    frequency: float = 1.0
    phase: float = 0.0
    offset: float = 1.0
    amplitude_jitter_std: float = 0.05
    frequency_jitter_std: float = 0.02
    phase_jitter_std: float = 0.10
    offset_jitter_std: float = 0.05


@dataclass
class NoiseConfig:
    mean: float = 0.5
    std: float = 0.2


@dataclass
class SnrConfig:
    epsilon: float = 1e-12
    db: bool = True
    window_radius: int = 1


@dataclass
class AnomalyConfig:
    single_channel_probability: float = 0.0005
    single_channel_duration_poisson_lambda: float = 3.0
    group_probability: float = 0.0001
    group_channel_fraction: float = 0.5
    group_duration_poisson_lambda: float = 8.0
    snr_scale: float = 0.1


@dataclass
class PlotConfig:
    output_path: str | None = "toy_timeseries_plot.png"
    start_index: int = 0
    end_index: int = 200
    channels: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class ToyTimeseriesConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    snr: SnrConfig = field(default_factory=SnrConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def load_config(path: str | Path) -> ToyTimeseriesConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}

    return ToyTimeseriesConfig(
        dataset=DatasetConfig(**(raw.get("dataset") or {})),
        signal=SignalConfig(**(raw.get("signal") or {})),
        noise=NoiseConfig(**(raw.get("noise") or {})),
        snr=SnrConfig(**(raw.get("snr") or {})),
        anomaly=AnomalyConfig(**(raw.get("anomaly") or {})),
        plot=PlotConfig(**(raw.get("plot") or {})),
    )


def make_timestamps(config: ToyTimeseriesConfig) -> np.ndarray:
    return np.arange(config.dataset.n_timesteps, dtype=float) * config.dataset.dt


def sample_channel_params(
    config: ToyTimeseriesConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n_channels = config.dataset.n_channels
    signal = config.signal

    return {
        "amplitude": signal.amplitude + rng.normal(0.0, signal.amplitude_jitter_std, n_channels),
        "frequency": signal.frequency + rng.normal(0.0, signal.frequency_jitter_std, n_channels),
        "phase": signal.phase + rng.normal(0.0, signal.phase_jitter_std, n_channels),
        "offset": signal.offset + rng.normal(0.0, signal.offset_jitter_std, n_channels),
    }


def make_clean_signal(
    timestamps: np.ndarray,
    channel_params: dict[str, np.ndarray],
) -> np.ndarray:
    n_timesteps = len(timestamps)
    n_channels = len(channel_params["amplitude"])
    clean = np.empty((n_timesteps, n_channels), dtype=float)

    for channel in range(n_channels):
        clean[:, channel] = (
            channel_params["amplitude"][channel]
            * np.sin(
                2.0 * np.pi * channel_params["frequency"][channel] * timestamps
                + channel_params["phase"][channel]
            )
            + channel_params["offset"][channel]
        )

    return clean


def make_anomaly_masks(
    shape: tuple[int, int],
    config: AnomalyConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    single_channel_mask = np.zeros(shape, dtype=bool)
    group_mask = np.zeros(shape, dtype=bool)
    n_timesteps, n_channels = shape
    n_group_channels = max(1, round(n_channels * config.group_channel_fraction))

    for index in range(n_timesteps):
        single_channels = np.flatnonzero(
            rng.random(n_channels) < config.single_channel_probability
        )
        mark_anomaly(
            single_channel_mask,
            index,
            single_channels,
            config.single_channel_duration_poisson_lambda,
            rng,
        )

        if rng.random() < config.group_probability:
            group_channels = rng.choice(n_channels, size=n_group_channels, replace=False)
            mark_anomaly(
                group_mask,
                index,
                group_channels,
                config.group_duration_poisson_lambda,
                rng,
            )

    return {
        "single_channel_anomaly_mask": single_channel_mask,
        "group_anomaly_mask": group_mask,
        "anomaly_mask": single_channel_mask | group_mask,
    }


def make_anomaly_mask(
    shape: tuple[int, int],
    config: AnomalyConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    return make_anomaly_masks(shape, config, rng)["anomaly_mask"]


def mark_anomaly(
    mask: np.ndarray,
    start: int,
    channels: np.ndarray,
    duration_poisson_lambda: float,
    rng: np.random.Generator,
) -> None:
    if len(channels) == 0:
        return

    duration = max(1, rng.poisson(duration_poisson_lambda))
    end = min(mask.shape[0], start + duration)
    mask[start:end, channels] = True


def make_noise(config: ToyTimeseriesConfig, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(
        loc=config.noise.mean,
        scale=config.noise.std,
        size=(config.dataset.n_timesteps, config.dataset.n_channels),
    )


def compute_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    epsilon: float,
    use_db: bool,
    window_radius: int,
) -> np.ndarray:
    if window_radius <= 0:
        signal_power = clean**2
        noise_power = noise**2
    else:
        signal_power = np.empty_like(clean)
        noise_power = np.empty_like(noise)
        for index in range(clean.shape[0]):
            start = max(0, index - window_radius)
            end = min(clean.shape[0], index + window_radius + 1)
            signal_power[index] = np.mean(clean[start:end] ** 2, axis=0)
            noise_power[index] = np.mean(noise[start:end] ** 2, axis=0)

    snr = signal_power / (noise_power + epsilon)
    if use_db:
        return 10.0 * np.log10(snr + epsilon)
    return snr


def generate_toy_timeseries(config: ToyTimeseriesConfig) -> dict[str, Any]:
    rng = np.random.default_rng(config.dataset.seed)

    timestamps = make_timestamps(config)
    channel_params = sample_channel_params(config, rng)
    clean = make_clean_signal(timestamps, channel_params)
    noise = make_noise(config, rng)
    noisy = clean + noise
    snr = compute_snr(clean, noise, config.snr.epsilon, config.snr.db, config.snr.window_radius)
    anomaly_masks = make_anomaly_masks(snr.shape, config.anomaly, rng)
    snr[anomaly_masks["anomaly_mask"]] *= config.anomaly.snr_scale

    data = np.stack([clean, noise, noisy, snr], axis=-1)
    feature_names = [
        "clean_signal",
        "noise",
        "noisy_signal",
        "SNR db" if config.snr.db else "SNR",
    ]

    return {
        "timestamps": timestamps,
        "data": data,
        "feature_names": feature_names,
        "channel_params": channel_params,
        **anomaly_masks,
    }


def save_npz(example: dict[str, Any], path: str | Path) -> None:
    channel_params = example["channel_params"]
    np.savez(
        path,
        timestamps=example["timestamps"],
        data=example["data"],
        anomaly_mask=example["anomaly_mask"],
        single_channel_anomaly_mask=example["single_channel_anomaly_mask"],
        group_anomaly_mask=example["group_anomaly_mask"],
        feature_names=np.array(example["feature_names"]),
        channel_amplitude=channel_params["amplitude"],
        channel_frequency=channel_params["frequency"],
        channel_phase=channel_params["phase"],
        channel_offset=channel_params["offset"],
    )


def save_plot(example: dict[str, Any], config: PlotConfig, path: str | Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError("plotting requires matplotlib: pip install matplotlib") from exc

    timestamps = example["timestamps"][config.start_index : config.end_index]
    data = example["data"][config.start_index : config.end_index]
    single_channel_mask = example["single_channel_anomaly_mask"][
        config.start_index : config.end_index, config.channels
    ]
    group_mask = example["group_anomaly_mask"][config.start_index : config.end_index, config.channels]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for channel in config.channels:
        axes[0].plot(timestamps, data[:, channel, 2], label=f"channel {channel}")
        axes[1].plot(timestamps, data[:, channel, 3], label=f"channel {channel}")

    single_channel_color = "tab:orange"
    group_color = "tab:red"
    single_channel_intervals = mask_to_time_intervals(timestamps, single_channel_mask.any(axis=1))
    group_intervals = mask_to_time_intervals(timestamps, group_mask.any(axis=1))
    for axis in axes:
        for start, end in single_channel_intervals:
            axis.axvspan(start, end, facecolor=single_channel_color, alpha=0.18, edgecolor="none")
        for start, end in group_intervals:
            axis.axvspan(start, end, facecolor=group_color, alpha=0.18, edgecolor="none")

    axes[0].set_title("Detector response")
    axes[0].set_ylabel("amplitude")
    axes[1].set_title(example["feature_names"][3])
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(example["feature_names"][3])

    anomaly_handles = []
    if single_channel_intervals:
        anomaly_handles.append(
            Patch(facecolor=single_channel_color, alpha=0.18, label="single-channel anomaly")
        )
    if group_intervals:
        anomaly_handles.append(Patch(facecolor=group_color, alpha=0.18, label="group anomaly"))

    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(handles + anomaly_handles, labels + [handle.get_label() for handle in anomaly_handles], loc="upper right")
        axis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mask_to_time_intervals(timestamps: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    if len(timestamps) == 0 or len(mask) == 0:
        return []

    intervals = []
    padded = np.concatenate(([False], mask, [False]))
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 1.0

    for start, end in zip(starts, ends, strict=True):
        intervals.append((float(timestamps[start]), float(timestamps[end - 1] + dt)))

    return intervals


def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    config = load_config(config_path)
    example = generate_toy_timeseries(config)

    print(f"data shape: {example['data'].shape}")
    print(f"features: {example['feature_names']}")

    if config.dataset.output_path:
        output_path = Path(config.dataset.output_path)
        if not output_path.is_absolute():
            output_path = config_path.parent / output_path
        save_npz(example, output_path)
        print(f"saved: {output_path}")

    if config.plot.output_path:
        plot_path = Path(config.plot.output_path)
        if not plot_path.is_absolute():
            plot_path = config_path.parent / plot_path
        try:
            save_plot(example, config.plot, plot_path)
            print(f"plotted: {plot_path}")
        except ImportError as exc:
            print(f"skipped plot: {exc}")


if __name__ == "__main__":
    main()
