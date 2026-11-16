# Bubbleformer rollout vs simulation, Twall 107

Frames 105-1999 (1895 frames, 189.5 time units). Rollout starts from simulation frames 100-104. Checkpoint: `Bubbleformer-S-PB-Subcooled.ckpt`.

| Statistic | Bubbleformer rollout | simulation Twall 107 | simulation, second half vs first half | simulation Twall 106 | simulation Twall 108 |
|---|---|---|---|---|---|
| departures | 136 | 222 | 121 | 208 | 232 |
| departures per 100 frames | 7.18 | 11.7 | 12.8 | 11 | 12.2 |
| mean interval (frames) | 13.9 | 8.43 | 7.59 | 9.08 | 8.15 |
| interval coefficient of variation | 0.731 | 0.79 | 0.777 | 0.698 | 0.8 |
| interval distance to simulation (frames) | 5.41 | 0 | 1.73 | 0.841 | 0.34 |
| heat-flux mean | 6.02e+04 | 6.06e+04 | 6.14e+04 | 5.99e+04 | 6.29e+04 |
| heat-flux std | 7.36e+03 | 7.64e+03 | 7.58e+03 | 6.84e+03 | 7.33e+03 |
| heat-flux histogram distance to simulation | 590 | 0 | 1.59e+03 | 1.01e+03 | 2.31e+03 |
| heat-flux autocorrelation RMS difference to simulation | 0.0568 | 0 | 0.0659 | 0.0416 | 0.0464 |
| heat flux around departures, RMS difference to simulation | 1.22e+03 | 0 | 1.56e+03 | 1.36e+03 | 1.88e+03 |
| vapor fraction | 0.0366 | 0.035 | 0.0369 | 0.0344 | 0.0411 |
