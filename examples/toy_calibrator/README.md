# Toy Time-Series Example

This example generates a simple synthetic multichannel time-series dataset in canonical
`(T, C, F)` form.

It is meant for demos, sanity checks, plotting, and downstream preprocessing tests. It is
not a realistic hardware simulator.

## Files

- `generate_toy_timeseries.py` — generator script and small helper functions.
- `config.yaml` — editable YAML configuration.
- `toy_timeseries.npz` — generated NumPy archive, created when the script runs.
- `toy_timeseries_plot.png` — generated plot, created when the script runs.

## Run

From the repository root:

```bash
python examples/toy_calibrator/generate_toy_timeseries.py
```

The script reads:

```text
examples/toy_calibrator/config.yaml
```

and writes output paths configured in that YAML.

## Output array

The main generated array is saved as `data` in the `.npz` file.

Shape:

```text
(T, C, F)
```

where:

- `T` = number of timesteps;
- `C` = number of channels;
- `F = 4` features.

Feature layout:

```text
data[:, :, 0] = clean_signal
data[:, :, 1] = noise
data[:, :, 2] = noisy_signal
data[:, :, 3] = SNR or SNR db
```

The `feature_names` array in the `.npz` stores the feature names in order.

The output archive also includes:

- `timestamps`: shape `(T,)`
- `anomaly_mask`: shape `(T, C)`, boolean mask where SNR anomalies were injected
- `channel_amplitude`: shape `(C,)`
- `channel_frequency`: shape `(C,)`
- `channel_phase`: shape `(C,)`
- `channel_offset`: shape `(C,)`

## Simulation summary

The simulation has four main parts:

1. create similar but not identical clean sine-wave channels;
2. add independent random Gaussian noise;
3. compute local/windowed SNR;
4. optionally inject rare low-SNR anomaly periods into the SNR feature only.

The raw signal response is not modified by anomalies:

```text
noisy_signal = clean_signal + noise
```

Anomalies only modify the saved SNR feature.

## Time axis

The timestamp vector is:

```python
timestamps = np.arange(n_timesteps) * dt
```

Configured by:

```yaml
dataset:
  n_timesteps: 10_000
  dt: 0.01
```

## Clean signal model

Each channel is a sine wave:

```text
clean[t, c] = A_c * sin(2π * f_c * timestamps[t] + phase_c) + offset_c
```

The channels are somewhat correlated because they share common base parameters. Each channel
gets a small random jitter around those base parameters.

For each channel:

```text
A_c      = amplitude + Normal(0, amplitude_jitter_std)
f_c      = frequency + Normal(0, frequency_jitter_std)
phase_c  = phase + Normal(0, phase_jitter_std)
offset_c = offset + Normal(0, offset_jitter_std)
```

This makes the channels similar without being exactly identical.

## Noise model

Noise is sampled independently for each timestep and channel:

```text
noise[t, c] ~ Normal(mean, std)
```

Configured by:

```yaml
noise:
  mean: 0.1
  std: 0.02
```

The noise is not channel-correlated. Any similarity between channels comes from the clean
signal model, not from the noise.

## Noisy signal

The observed/noisy signal is:

```text
noisy_signal = clean_signal + noise
```

This relationship is preserved even when SNR anomalies are injected.

## SNR model

SNR is computed from known synthetic truth:

```text
SNR = signal_power / (noise_power + epsilon)
```

If `window_radius` is `0`, the power is pointwise:

```text
signal_power[t, c] = clean[t, c]^2
noise_power[t, c]  = noise[t, c]^2
```

If `window_radius` is greater than `0`, local/windowed power is used. For example:

```yaml
snr:
  window_radius: 1
```

means each middle timestep uses 3 total points:

```text
left neighbor + current point + right neighbor
```

The local power is:

```text
signal_power[t, c] = mean(clean[t-1:t+2, c]^2)
noise_power[t, c]  = mean(noise[t-1:t+2, c]^2)
```

At the edges, the window is clipped to valid indices.

If `db` is `true`, the final SNR feature is converted to decibels:

```text
snr_db = 10 * log10(snr_linear + epsilon)
```

If `db` is `false`, the saved SNR feature remains linear.

## SNR anomalies

Anomalies simulate periods where hardware quality drops. In this example, anomalies are
injected only into the SNR feature.

They do not change:

- `clean_signal`
- `noise`
- `noisy_signal`

An anomaly marks a time/channel region in `anomaly_mask`, then scales the SNR value:

```text
snr[anomaly_mask] *= snr_scale
```

For example:

```yaml
anomaly:
  snr_scale: 0.1
```

means affected SNR values drop to 10% of their original value.

### Single-channel anomalies

Single-channel anomalies are independent per channel and timestep.

Configured by:

```yaml
anomaly:
  single_channel_probability: 0.0005
  single_channel_duration_poisson_lambda: 3
```

At each timestep, each channel has probability `single_channel_probability` of starting a
single-channel anomaly.

The duration is sampled as:

```python
duration = max(1, rng.poisson(single_channel_duration_poisson_lambda))
```

So the duration is Poisson-distributed but never shorter than one timestep.

### Group-channel anomalies

Group anomalies simulate a larger hardware event affecting a fraction of channels at once.

Configured by:

```yaml
anomaly:
  group_probability: 0.0001
  group_channel_fraction: 0.5
  group_duration_poisson_lambda: 8
```

At each timestep, there is probability `group_probability` of starting a group anomaly.

When one starts:

1. `round(n_channels * group_channel_fraction)` channels are selected randomly;
2. the selected channels are marked anomalous;
3. duration is sampled as:

```python
duration = max(1, rng.poisson(group_duration_poisson_lambda))
```

Single-channel and group anomalies can overlap. The saved `anomaly_mask` is boolean, so an
affected point is either anomalous or not. Overlapping anomalies do not compound the SNR
scale.

## Plotting

The script saves a PNG with two panels:

1. noisy signal;
2. SNR feature.

Configured by:

```yaml
plot:
  output_path: toy_timeseries_plot.png
  start_index: 0
  end_index: 1000
  channels: [0, 1, 2]
```

This plots the selected channels over the configured index range.

## Full YAML configuration

Current example configuration:

```yaml
dataset:
  n_timesteps: 10_000
  n_channels: 8
  dt: 0.01
  seed: 123
  output_path: toy_timeseries.npz

signal:
  amplitude: 0.1
  frequency: 1.0
  phase: 0.0
  offset: 1.2

  amplitude_jitter_std: 0.005
  frequency_jitter_std: 0.002
  phase_jitter_std: 0.01
  offset_jitter_std: 0.005

noise:
  mean: 0.1
  std: 0.02

snr:
  epsilon: 1.0e-8
  db: false
  window_radius: 1

anomaly:
  single_channel_probability: 0.0005
  single_channel_duration_poisson_lambda: 3

  group_probability: 0.0001
  group_channel_fraction: 0.5
  group_duration_poisson_lambda: 8

  snr_scale: 0.1

plot:
  output_path: toy_timeseries_plot.png
  start_index: 0
  end_index: 1000
  channels: [0, 1, 2]
```

## Parameter reference

### `dataset`

| Parameter | Meaning |
|---|---|
| `n_timesteps` | Number of time samples, `T`. |
| `n_channels` | Number of channels, `C`. |
| `dt` | Time spacing between samples. |
| `seed` | Random seed for reproducible output. Use `null` for nondeterministic output. |
| `output_path` | `.npz` output path. Relative paths are resolved under `examples/toy_calibrator/`. Use `null` to skip saving. |

### `signal`

| Parameter | Meaning |
|---|---|
| `amplitude` | Base sine amplitude shared by channels. |
| `frequency` | Base sine frequency shared by channels. |
| `phase` | Base sine phase shared by channels. |
| `offset` | Base vertical offset of the clean signal. |
| `amplitude_jitter_std` | Per-channel Gaussian jitter std for amplitude. |
| `frequency_jitter_std` | Per-channel Gaussian jitter std for frequency. |
| `phase_jitter_std` | Per-channel Gaussian jitter std for phase. |
| `offset_jitter_std` | Per-channel Gaussian jitter std for signal offset. |

### `noise`

| Parameter | Meaning |
|---|---|
| `mean` | Mean of additive Gaussian noise. |
| `std` | Standard deviation of additive Gaussian noise. |

### `snr`

| Parameter | Meaning |
|---|---|
| `epsilon` | Small value added to avoid division by zero and log of zero. |
| `db` | If `true`, save SNR in decibels. If `false`, save linear SNR. |
| `window_radius` | Number of neighbors on each side used for local SNR power. `0` means pointwise SNR. `1` means 3-point local SNR. |

### `anomaly`

| Parameter | Meaning |
|---|---|
| `single_channel_probability` | Per-timestep, per-channel probability of starting a single-channel SNR anomaly. |
| `single_channel_duration_poisson_lambda` | Poisson lambda for single-channel anomaly duration. Minimum duration is one timestep. |
| `group_probability` | Per-timestep probability of starting a group-channel SNR anomaly. |
| `group_channel_fraction` | Fraction of channels affected by a group anomaly. |
| `group_duration_poisson_lambda` | Poisson lambda for group anomaly duration. Minimum duration is one timestep. |
| `snr_scale` | Multiplicative scale applied to SNR during anomalies. `0.1` means SNR drops to 10%. |

### `plot`

| Parameter | Meaning |
|---|---|
| `output_path` | PNG output path. Relative paths are resolved under `examples/toy_calibrator/`. Use `null` to skip plotting. |
| `start_index` | First timestep index to plot. |
| `end_index` | Stop timestep index for plotting. Follows normal Python slicing, so this index is excluded. |
| `channels` | Channel indices to draw in the plot. |

## Loading the generated `.npz`

Example:

```python
import numpy as np

with np.load("examples/toy_calibrator/toy_timeseries.npz") as z:
    data = z["data"]
    timestamps = z["timestamps"]
    anomaly_mask = z["anomaly_mask"]
    feature_names = z["feature_names"]

clean = data[:, :, 0]
noise = data[:, :, 1]
noisy = data[:, :, 2]
snr = data[:, :, 3]
```

## Notes and limitations

- This is a small example, not a validated simulator.
- YAML values are loaded directly into dataclasses. Typos generally fail with normal Python
  `TypeError` messages.
- No strict validation is performed for probabilities, positive durations, valid channel
  indices, or valid plot ranges.
- SNR anomalies are synthetic labels/diagnostics. They are not caused by a change in the raw
  clean or noisy signal.
