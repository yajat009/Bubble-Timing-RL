# Bubbleformer rollout vs simulation, Twall 97

Frames 105-1999 (1895 frames, 189.5 time units). Rollout starts from simulation frames 100-104. Checkpoint: `Bubbleformer-S-PB-Subcooled.ckpt`.

| Statistic | Bubbleformer rollout | simulation Twall 97 | sim, 2nd half vs 1st half | simulation Twall 96 | simulation Twall 98 |
|---|---|---|---|---|---|
| departures | 28 | 157 | 82 | 149 | 191 |
| departures per 100 frames | 1.48 | 8.28 | 8.65 | 7.86 | 10.1 |
| mean interval (frames) | 68.3 | 12 | 11.5 | 12.8 | 9.86 |
| interval CV | 1.82 | 0.801 | 0.874 | 0.821 | 0.814 |
| interval W1 to sim (frames) | 46.3 | 0 | 2.08 | 1.01 | 2.14 |
| heat flux mean | 4.42e+04 | 4.73e+04 | 4.72e+04 | 4.64e+04 | 4.97e+04 |
| heat flux std | 6.73e+03 | 5.8e+03 | 5.75e+03 | 5.54e+03 | 6.21e+03 |
| heat flux histogram W1 to sim | 3.1e+03 | 0 | 409 | 909 | 2.48e+03 |
| heat flux ACF RMS diff to sim | 0.461 | 0 | 0.119 | 0.0693 | 0.113 |
| event response RMS diff to sim | 2.34e+03 | 0 | 840 | 1.34e+03 | 2.13e+03 |
| vapor fraction | 0.0225 | 0.0263 | 0.0278 | 0.0258 | 0.0288 |
