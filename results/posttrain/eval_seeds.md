# Evaluation spread over seeds (coarse 128 x 128 statistics)

Mean +- standard deviation over seeds of one 1,895-frame rollout each. departure_rate is per 100 frames; the other columns are distances to the simulation (0 = same).

## Twall 97 (simulation: 7.91 departures per 100 frames)

| Head | Seeds | departure_rate | waiting_times | quiet_periods | heat_flux_histogram | vapor_fraction |
|---|---|---|---|---|---|---|
| S | 1 | 1.53 +- 0 | 43.5 +- 0 | 39.1 +- 0 | 3.1e+03 +- 0 | 0.00322 +- 0 |
| head | 4 | 2.94 +- 0.41 | 20.3 +- 4.4 | 12.7 +- 6.6 | 4.74e+03 +- 5e+03 | 0.00447 +- 0.0014 |
| R2 | 4 | 3.42 +- 0.4 | 16.8 +- 3.6 | 6.85 +- 1.6 | 1.41e+04 +- 1.2e+04 | 0.00335 +- 0.002 |
| R3 | 4 | 3.28 +- 0.45 | 17.3 +- 4.8 | 7.61 +- 3.3 | 1.2e+04 +- 2.1e+03 | 0.00572 +- 0.0023 |

## Twall 107 (simulation: 12.13 departures per 100 frames)

| Head | Seeds | departure_rate | waiting_times | quiet_periods | heat_flux_histogram | vapor_fraction |
|---|---|---|---|---|---|---|
| S | 1 | 7.28 +- 0 | 5.55 +- 0 | 4.37 +- 0 | 567 +- 0 | 0.00188 +- 0 |
| head | 4 | 6.68 +- 0.58 | 6.91 +- 1.4 | 2.9 +- 1.1 | 2.69e+03 +- 7.9e+02 | 0.00841 +- 0.0017 |
| R2 | 4 | 6.36 +- 1.1 | 8.04 +- 3 | 3.92 +- 2.2 | 4.23e+03 +- 1.9e+03 | 0.00678 +- 0.0044 |
| R3 | 4 | 9.2 +- 0.86 | 2.69 +- 0.82 | 3.61 +- 1.7 | 5.27e+03 +- 6.1e+03 | 0.00658 +- 0.0039 |

