# Bubbleformer rollouts from several start frames, Twall 97

Up to 1000 predicted frames per start. Checkpoint: `Bubbleformer-S-PB-Subcooled.ckpt`.

| Statistic | start 250: sim | start 250: rollout | start 500: sim | start 500: rollout | start 750: sim | start 750: rollout | start 995: sim | start 995: rollout |
|---|---|---|---|---|---|---|---|---|
| departures | 77 | 26 | 70 | 16 | 75 | 28 | 85 | 14 |
| departures per 100 frames | 7.7 | 2.6 | 7 | 1.6 | 7.5 | 2.8 | 8.5 | 1.4 |
| mean interval (frames) | 13 | 39.5 | 13.7 | 58.7 | 13.4 | 36.9 | 11.7 | 67.2 |
| interval CV | 0.694 | 1.07 | 0.723 | 1.09 | 0.814 | 0.771 | 0.879 | 1.18 |
| interval W1 to sim (frames) | - | 24.2 | - | 40.6 | - | 23.1 | - | 48.8 |
| heat flux mean | 4.61e+04 | 4.78e+04 | 4.58e+04 | 4.75e+04 | 4.63e+04 | 4.57e+04 | 4.71e+04 | 5.4e+04 |
| vapor fraction | 0.0258 | 0.0214 | 0.0268 | 0.0167 | 0.0275 | 0.0268 | 0.0277 | 0.0219 |
