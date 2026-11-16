# Bubbleformer short-term accuracy, Twall 97

20 start frames from 100 to 1985. Checkpoint: `Bubbleformer-S-PB-Subcooled.ckpt`. Relative L2 error (lower is better) and vapor IoU (higher is better), mean over starts. Persistence = repeat the last input frame.

| Span | Predictor | rel L2 dfun | rel L2 temperature | rel L2 velx | rel L2 vely | vapor IoU |
|---|---|---|---|---|---|---|
| frame 1 | Bubbleformer | 0.009 | 0.009 | 0.121 | 0.097 | 0.875 |
| frame 1 | persistence | 0.024 | 0.014 | 0.332 | 0.263 | 0.756 |
| frames 1-5 | Bubbleformer | 0.017 | 0.011 | 0.173 | 0.142 | 0.851 |
| frames 1-5 | persistence | 0.055 | 0.025 | 0.471 | 0.380 | 0.537 |
| frames 6-10 | Bubbleformer | 0.032 | 0.017 | 0.271 | 0.224 | 0.761 |
| frames 6-10 | persistence | 0.119 | 0.038 | 0.643 | 0.522 | 0.291 |
