# Locally Simultaneous Inference

This repository contains simulation code accompanying the paper [*Locally Simultaneous Inference*](https://arxiv.org/abs/2212.09009) by Zrnic and Fithian (2023).

The code implements locally simultaneous inference and several baselines: simultaneous inference methods, including PoSI by Berk et al. (2013), conditional inference due to Lee et al. (2016), and hybrid inference due to Andrews et al. (2019) and McCloskey (2020).

The repository contains notebooks with examples:
- inference on the winner and inference on all observations that exceed a threshold  ([```most-promising-effects.ipynb```](https://github.com/tijana-zrnic/locally-simultaneous-inference/blob/main/most-promising-effects.ipynb))
- inference after model selection via the LASSO ([```lasso.ipynb```](https://github.com/tijana-zrnic/locally-simultaneous-inference/blob/main/lasso.ipynb))
- inference on most extreme days/locations on global climate data ([```climate-data.ipynb```](https://github.com/tijana-zrnic/locally-simultaneous-inference/blob/main/climate-data.ipynb))
