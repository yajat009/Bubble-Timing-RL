# Bubbleformer rollout vs simulation, Twall 97

Frames 105-1999 (1895 frames, 189.5 time units). Rollout starts from simulation frames 100-104. Checkpoint: `Bubbleformer-S-PB-Subcooled.ckpt`.

| Statistic | Bubbleformer rollout | simulation Twall 97 | simulation, second half vs first half | simulation Twall 96 | simulation Twall 98 |
|---|---|---|---|---|---|
| departures | 47 | 157 | 82 | 149 | 191 |
| departures per 100 frames | 2.48 | 8.28 | 8.65 | 7.86 | 10.1 |
| mean interval (frames) | 40.8 | 12 | 11.5 | 12.8 | 9.86 |
| interval coefficient of variation | 1.38 | 0.801 | 0.874 | 0.821 | 0.814 |
| interval distance to simulation (frames) | 26.6 | 0 | 2.08 | 1.01 | 2.14 |
| heat-flux mean | 4.64e+04 | 4.73e+04 | 4.72e+04 | 4.64e+04 | 4.97e+04 |
| heat-flux std | 5.55e+03 | 5.8e+03 | 5.75e+03 | 5.54e+03 | 6.21e+03 |
| heat-flux histogram distance to simulation | 859 | 0 | 409 | 909 | 2.48e+03 |
| heat-flux autocorrelation RMS difference to simulation | 0.182 | 0 | 0.119 | 0.0693 | 0.113 |
| heat flux around departures, RMS difference to simulation | 1.77e+03 | 0 | 840 | 1.34e+03 | 2.13e+03 |
| vapor fraction | 0.0195 | 0.0263 | 0.0278 | 0.0258 | 0.0288 |
