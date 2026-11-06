# Video Frame Interpolation using Optical Flow (UCF101)

End-semester project: Image Analysis & Computer Vision.

Given two consecutive video frames, this project synthesises the frame(s) in between using **dense optical flow**
(classical computer vision, no deep learning, no GPU needed). It is evaluated on the **UCF101** action dataset by
hiding a real frame, re-creating it, and measuring PSNR / SSIM against the real one.

The base paper is "Phase-Based Frame Interpolation for Video" (Meyer et al., CVPR 2015). This project keeps the
topic but replaces the phase-based method with optical flow (DIS and Farneback) and occlusion-aware blending.

---

## Quick start (Windows PowerShell)

1. Unzip the project and open a terminal (PowerShell) in the folder that contains this README.
   Everything below is run from inside the project folder:

```powershell
cd Optical_Flow_Occlusion
```

2. Create a virtual environment and install the packages (first time only):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

   Python 3.9 or newer is needed. When the environment is active, the prompt starts with `(venv)`.
   (Mac/Linux: `source venv/bin/activate`.)

3. Start the web interface:

```powershell
streamlit run app.py
```

   Your browser opens at http://localhost:8501. If it does not, open that address by hand.
   In the page: upload a video (or pick one from `UCF-101`), choose the flow method (`dis`, `farneback` or
   `fused`), and use the tabs to compare methods, see the optical-flow arrows, make a slow-motion video, and view
   the saved results. Press `Ctrl+C` in the terminal to stop it.

Next time you only need:

```powershell
cd Optical_Flow_Occlusion
venv\Scripts\activate
streamlit run app.py
```

> The `venv` folder is only the installed packages. It is not part of the project and should not be zipped or
> submitted. Anyone can re-create it with the commands in step 2.

---

## 1. Dataset layout

Full UCF101 can be downloaded from https://www.crcv.ucf.edu/data/UCF101.php.
Because of limited disk space, this copy of the project keeps only three action classes:

```
Optical_Flow_Occlusion/
  UCF-101/
    ApplyEyeMakeup/
    ApplyLipstick/
    Archery/
  flowinterp/    app.py    evaluate.py    ...
```

All commands below use `--data_dir UCF-101` (the folder inside the project).

> **Note on the numbers in the report.** The results in the report (PSNR / SSIM tables, motion groups, the Diving
> and ApplyLipstick stills) were produced earlier from 200 videos sampled at random from all 101 UCF101 classes.
> With only three classes the sampled videos are different, so re-running the evaluation gives similar trends but
> not identical numbers, and the Diving examples cannot be recreated. Keep the original `results_gap1`,
> `results_gap2`, `results_gap3` folders if you still have them, since the report quotes those.

## 2. Other ways to run the project

Run these from inside `Optical_Flow_Occlusion` with the environment active (`venv\Scripts\activate`).

### a) Quick smoke test (no dataset needed)
```powershell
python make_sample_data.py --out sample_data --videos 6
python evaluate.py --data_dir sample_data --out_dir results_test --num_videos 0
```

### b) Evaluation on UCF101 (three classes)
```powershell
python evaluate.py --data_dir UCF-101 --num_videos 200 --triplets_per_video 3 --gap 1 --out_dir results_gap1
python evaluate.py --data_dir UCF-101 --num_videos 200 --triplets_per_video 3 --gap 2 --out_dir results_gap2
python evaluate.py --data_dir UCF-101 --num_videos 200 --triplets_per_video 3 --gap 3 --out_dir results_gap3
```
A larger gap means the frames are further apart, so there is more motion and it is harder. If the three folders hold
fewer than 200 videos in total, the script uses all of them; `--num_videos 0` evaluates every video.
Roughly 0.2 s per triplet on a laptop CPU. `--seed` (default 0) makes the sampled videos reproducible.

Outputs in each results folder:

| file | content |
|---|---|
| `summary.md` | mean PSNR / SSIM for every method, win rate, motion groups, significance tests |
| `metrics_bar.png`, `psnr_gain_hist.png` | figures |
| `per_triplet_results.csv`, `per_class_psnr.csv` | raw numbers, per action class |
| `samples/*.png` | example grids: frame A, ground truth, frame B, cross-fade, flow result, error x4 |

### c) Still images of the highest-motion moments of a clip
```powershell
python montage.py --input "UCF-101\ApplyLipstick\v_ApplyLipstick_g01_c01.avi" --top 3 --gap 2
```

### d) Running-time benchmark
```powershell
python benchmark_runtime.py
```

### e) Slow-motion video on any clip (also works on your own video)
```powershell
python demo.py --input "UCF-101\Archery\v_Archery_g01_c01.avi" --factor 4 --compare
```
Writes the 4x-frame-rate video, a side-by-side comparison video (original | cross-fade | optical flow) and a
flow-visualisation image to `demo_output/`.

### f) Extra experiments (optional)
```powershell
python extras.py --help             # ablation, occlusion strength, brightness change
python fusion_experiment.py --help  # per-pixel combination of DIS and Farneback
```
These are experimental and are not reported as results in the project report.

## 3. Method

For frames I0, I1 and time t in (0,1):

1. **Optical flow** both ways, F01 and F10, with **DIS** (Dense Inverse Search, Kroeger et al. 2016) or
   **Farneback** (polynomial expansion, 2003). Both are in OpenCV.
2. **Intermediate flow approximation** (Jiang et al., Super SloMo 2018):
   `F_t0 = -(1-t)t F01 + t² F10`, `F_t1 = (1-t)² F01 - t(1-t) F10`.
3. **Backward warping** of I0 and I1 to time t with `cv2.remap` (bilinear).
4. **Occlusion handling** with a forward-backward consistency check
   `e(x) = |F01(x) + F10(x + F01(x))|`. Pixels with large `e` are probably hidden in the other frame and get a
   lower weight `V = exp(-α e)` (α = 0.5, set by hand).
5. **Blending**: `It = ((1-t)V0·W0 + t·V1·W1) / ((1-t)V0 + t·V1)`.

Compared methods: `average` (cross-fade baseline), `flow_oneway_dis` (warp one frame only), `dis` and
`farneback` (full method). Metrics: PSNR and SSIM (scikit-image).

## 4. Main results reported (200 videos x 3 triplets = 600 per gap, DIS flow vs cross-fade)

| Gap | PSNR cross-fade | PSNR DIS flow | Win rate |
|---|---|---|---|
| 1 | 32.59 dB | 33.66 dB | 61.7% |
| 2 | 28.10 dB | 29.13 dB | 65.5% |
| 3 | 26.12 dB | 26.68 dB | 58.8% |

Flow helps most for small to moderate motion, gives no gain for almost still scenes, and its advantage fades for
the largest motions. The phase-based method of the base paper was not re-implemented, so only trends (not
absolute numbers) can be compared with the paper.

## 5. Files

```
flowinterp/core.py        flow estimation, warping, consistency check, interpolation, baselines
evaluate.py               drop-a-frame evaluation: PSNR/SSIM, tables, plots, sample images
montage.py                still-image comparison grids for the highest-motion moments of a clip
benchmark_runtime.py      seconds per interpolated frame at several frame sizes
demo.py                   slow-motion video from any clip
app.py, ui_logic.py       Streamlit web interface and its helper functions
fusion.py                 per-pixel combination of DIS and Farneback (experimental)
extras.py                 ablation / occlusion-strength / brightness experiments
fusion_experiment.py      evaluation of the fused method
make_sample_data.py       tiny synthetic dataset for smoke-testing
requirements.txt          Python packages
UCF-101/                  three UCF101 classes (ApplyEyeMakeup, ApplyLipstick, Archery)
results_gap1..3/          evaluation outputs
```

## 6. Troubleshooting

* `streamlit is not recognized` or `ModuleNotFoundError`: the environment is not active or the packages are not
  installed. Run `venv\Scripts\activate`, then `pip install -r requirements.txt`.
* `running scripts is disabled on this system` when activating: run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that terminal, then activate again.
* `Cannot open video` / empty frames: the OpenCV build lacks the XVID codec for `.avi`. Use
  `pip install opencv-python` (not `-headless`), or re-encode with `ffmpeg -i in.avi out.mp4`.
* `module 'cv2' has no attribute 'DISOpticalFlow_create'`: use `cv2.DISOpticalFlow.create(...)` in
  `flowinterp/core.py`, or change the OpenCV version.
* A video in the web page will not play: install `imageio-ffmpeg` (`pip install imageio-ffmpeg`).
* Slow runs: lower `--num_videos` or `--triplets_per_video`.
* `__pycache__` folders are created automatically and can be deleted.