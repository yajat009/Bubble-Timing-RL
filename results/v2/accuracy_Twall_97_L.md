# Bubbleformer short-term accuracy, Twall 97

5 start frames from 100 to 1985. Checkpoint: `Bubbleformer-L-PB-Subcooled.ckpt`. Relative L2 error (lower is better) and vapor IoU (higher is better), mean over starts. Persistence = repeat the last input frame.

| Span | Predictor | rel L2 dfun | rel L2 temperature | rel L2 velx | rel L2 vely | vapor IoU |
|---|---|---|---|---|---|---|
| frame 1 | Bubbleformer | 0.009 | 0.008 | 0.120 | 0.098 | 0.883 |
| frame 1 | persistence | 0.032 | 0.014 | 0.338 | 0.288 | 0.787 |
| frames 1-5 | Bubbleformer | 0.014 | 0.010 | 0.170 | 0.149 | 0.869 |
| frames 1-5 | persistence | 0.048 | 0.024 | 0.462 | 0.407 | 0.583 |
| frames 6-10 | Bubbleformer | 0.017 | 0.016 | 0.279 | 0.240 | 0.799 |
| frames 6-10 | persistence | 0.086 | 0.037 | 0.628 | 0.541 | 0.351 |
