<!-- HEADER -->
<p align="center">
    <h1 align="center">MoLingo: Motion-Language Alignment for Text-to-Motion Generation</h1>
    <!-- authors -->
    <p align="center">
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/He.html"><b>Yannan He</b></a>
        &emsp;
        <a href="https://garvita-tiwari.github.io"><b>Garvita Tiwari</b></a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/Zhang.html"><b>Xiaohan Zhang</b></a>
        &emsp;
        <a href="https://www.linkedin.com/in/pankaj-bora-2045891b5/?originalSubdomain=de"><b>Pankaj Bora</b></a>
    </p>
   <p align="center">
        <a href="https://tolgabirdal.github.io"><b>Tolga Birdal</b></a>
        &emsp;
        <a href="https://janericlenssen.github.io"><b>Jan Eric Lenssen</b></a>
        &emsp;
        <a href="https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html"><b>Gerard Pons-Moll</b></a>
    </p>
    <!-- conference -->
    <h3 align="center">CVPR 2026</h3>
    <!-- teaser -->
    <p align="center">
        <img src="assets/teaser.gif" alt="Project Teaser" width="600px">
    </p>
    <!-- badges -->
    <p align="center">
        <a href="https://arxiv.org/abs/2512.13840">
            <img src="https://img.shields.io/badge/arXiv-2512.13840-b31b1b.svg?style=for-the-badge" alt="Paper PDF">
        </a>
        &emsp;
        <a href="https://hynann.github.io/molingo/MoLingo.html">
            <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=Google%20chrome&logoColor=white" alt="Project Page">
        </a>
    </p>
</p>


## News

- [2026-03-07] **Note: We have updated the pre-trained 272-dimensional model and its SAE with better checkpoints. If you downloaded the version from the initial commit, please run ```prepare/download_models.sh``` again to get the latest version.**
- [2026-03-07] Motion generation demo released, **pull the latest version and give it a try!**
- [2026-02-21] MoLingo is accepted at CVPR 2026!
- [2026-02-16] Evaluation scripts released
- [2025-12-15] Publish the paper on arXiv


## TODO
- [x] Release the evaluation pipeline
- [x] Release the motion generation pipeline
- [ ] Release the training script for the SAE
- [ ] Release the training script for the MoLingo model
- [ ] Release the G1 tracking pipeline


## Get You Ready

<details>

### 1. Conda Environment
```
conda env create -f environment.yml
conda activate molingo
```
We test our code on Python 3.10.13, PyTorch 2.9.0, and CUDA 12.8

### 2. Get Data

You have two options here:
* **Skip getting data**, if you just want to generate motions using *own* descriptions.
* **Get full data**, if you want to *re-train* and *evaluate* the model.

**(a) Original HumanML3D (263 dim)**

Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the result dataset to your own `data_root`

**(b) HumanML3D-272 (272 dim)**

Follow the instruction in [MotionStreamer](https://github.com/zju3dv/MotionStreamer), download the processed 272-dim HumanML3D dataset (not BABEL), store in your own `data_root`

After processing, the directory structure:

```
data_root
├── HumanML3D
│   ├── HumanML3D
│   │   └── new_joint_vecs
│   │   └── new_joints
│   │   └── pose_data
│   │   └── train.txt
│   │   └── val.txt
│   │   └── test.txt
│   │   └── Mean.npy
│   │   └── Std.npy
│   ├─ ...
├── HumanML3D_272
│   ├── mean_std
│   ├── motion_data
│   ├── split
│   ├── texts
```

### 3. Models and Dependencies

#### Download Evaluation Models and Gloves

```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Set up TMR-263 evaluator

**(Not required unless you are evaluating with the TMR-263 evaluator.)**

Follow the instruction in [TMR](https://github.com/Mathux/TMR) (Installation - Set up the datasets), then copy the result dataset to directory `mogen/checkpoints/TMR`

After processing, the directory structure:
 ```
TMR
├── datasets
│   ├── annotations
│   │   └── humanml3d
│   │   └── ...
│   ├── motions
│   │   └── guo3dfeats
│   │   └── ...
├── models
│   ├── tmr_humanml3d_guoh3dfeats
│   │   └── contrastive_metrics
│   │   └── last_weights
│           ├── motion_decoder.pt
│           ├── motion_encoder.pt
│           ├── text_encoder.pt
│   │   └── latents
│   │   └── ...
│   ├── ...
├── stats
│   ├── humanml3d
│   │   └── guo3dfeats
│           ├── mean.pt
│           ├── std.pt
│   ├── ...
    
 ```

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### 

</details>

## Demo

<details>

```
python mogen/demo.py -a 1 -i assets/example.txt -b {your_smpl_model_path}
```

### Notes:

* ```-a``` denotes the acceleration ratio, meaning that ```a``` latents are sampled in each sampling iteration.
* An example of prompt file is given in ```./assets/example.txt```. Please follow the format of ```<text description>#<motion duration in seconds>``` at each line. The generated motions are in 30 fps.
* If you write ```<text description>#NA```, we will call the length estimator from  [MoMask](https://github.com/EricGuo5513/momask-codes) to determine a length. Note once there is one NA, all the others will be NA automatically.
* To render the animation video, we first apply FK to obtain the body joints. Be sure to specify the SMPL model path with ```-b```, replacing it with your own path. Follow [this](https://github.com/vchoutas/smplx#downloading-the-model) link for the instruction on setting it up.
 The ```-b``` should have [this](https://github.com/vchoutas/smplx#model-loading) structure.

### Some thoughts:

* We provide the motion generation script only for the 272-dimensional model, as we recommend using this improved representation. It allows us to directly extract the rotation component from the output, avoiding potential errors introduced by IK.
* This 272D model uses a temporal downscaling rate of ```2×``` and a ```32-d``` latent dimension (instead of ```4x16```). We found that, for the 272D model, finer temporal resolution and a richer latent space significantly improve motion quality.
* By default, we set the ```acc``` parameter to ```1``` to achieve the best practical motion generation quality. You can increase it for faster generation in scenarios such as evaluation.
* We still conduct standard evaluation on the HumanML3D-based representation with a ```4×16``` latent size to ensure fair comparison with existing methods, as discussed in our paper and the next ```Evaluation``` section.

</details>

## Evaluation

<details>

### Evaluate the 263-dim model with TMR-263 and MARDM-67 evaluator:

```
python mogen/eval_mogen.py -d 263 -c 5.5 -a 3 -r 20 -dr {your_data_root}
```

### Evaluate the 272-dim model with MS-272 evaluator:

```
python mogen/eval_mogen.py -d 272 -c 7.0 -a 5 -r 20 -dr {your_data_root}
```
#### 

</details>

## Acknowledgments

This code is standing on the shoulders of giants, we would like to thank the following contributors that our code is based on:

[MAR](https://github.com/LTH14/mar/), [TMR](https://github.com/Mathux/TMR/),  [rectified-flow](https://github.com/lqiang67/rectified-flow/tree/main), [MoMask](https://github.com/EricGuo5513/momask-codes), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [MotionStreamer](https://github.com/zju3dv/MotionStreamer), [272-dim-Motion-Representation](https://github.com/Li-xingXiao/272-dim-Motion-Representation), [MARDM](https://github.com/neu-vi/MARDM)


## Citation

If you find our code or paper helpful, please consider starring our repository and citing:
```
@inproceedings{he2026molingo,
      title={MoLingo: Motion–Language Alignment for Text-to-Human Motion Generation},
      author={He, Yannan and Tiwari, Garvita and Zhang, Xiaohan and Bora, Pankaj and Birdal, Tolga and Lenssen, Jan Eric and Pons-Moll, Gerard},
      booktitle={CVPR},
      year={2026}
  }
```