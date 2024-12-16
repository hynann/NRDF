## SMPL and SMPL+H model
The project uses [smplx](https://github.com/vchoutas/smplx) package to work with SMPL body model.
Follow [this](https://github.com/vchoutas/smplx#downloading-the-model) link for the instruction on setting it up.
After installing smplx make sure to set `bm_path=<path to folder with SMPL and SMPL+H models>` in configs to point to the folder with the [following](https://github.com/vchoutas/smplx#model-loading) 
structure. [Preprocessing](https://github.com/vchoutas/smplx/blob/main/tools/README.md#smpl-h-version-used-in-amass) SMPL+H models is necessary.

## Data preparation

To specify the folder structure of data used for this project, let's create a clear and structured layout. 

Assuming you have a main data directory `DATA_ROOT`, here is the desired folder structure after completing the following preparation steps:

```bash
<DATA_ROOT>
├── RAW_DATA
├── SAMPLED_POSES
├── FAISS_MODEL
├── NOISY_POSES
```

#### Folder descriptions:

  * `RAW_DATA/`: Contains the raw AMASS motion sequences
  * `SAMPLED_POSES/`: Location for storing the discrete poses sampled from temporal motion data
  * `FAISS_MODEL/`: Pretrained FAISS model for nearest neighbor searching
  * `NOISY_POSES/`: Location for storing all noisy poses along with their corresponding ground truth distance to the manifold. It serves as the main directory used for training

To prepare the data follow the steps below:

### 1. Downloading AMASS data

Manually create `RAW_DATA` folder under `<DATA_ROOT>`.

Register and download data from the official [website](https://amass.is.tue.mpg.de/). 

**Required splits**: ACCAD, BMLhandball, BMLmovi, BMLrub, CMU, EKUT, EyesJapanDataset, KIT, PosePrior, TCDHands, TotalCapture, SSM and Transitions


**Folder structure of the raw data:**
```bash
<DATA_ROOT>
├── RAW_DATA
│   ├── ACCAD
│   │   ├── Female1General_c3d
│   │   │   ├── A1-Stand_poses.npz
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── s011
│   ├── BioMotionLab_NTroje
│   │   ├── rub001
│   │   │   ├── 0000_treadmill_norm_poses.npz
│   │   │   ├── ...
│   │   ├── ...
│   │   ├── rub115
│   ├── ...
```


### 2. Sampling poses from AMASS


Run:

  ```bash
  python lib/data/sample_amass.py -d <DATA_ROOT> -m train -r 0.3
  ```

**Notes:**

  * `-d` specifies the directory where the raw data and sampled poses are stored

  * `-m` controls train/val/test split of the AMASS dataset

  * `-r` adjusts the rate for temporal motion sampling


**Folders structure after sampling:**

```bash
<DATA_ROOT>
├── RAW_DATA
├── SAMPLED_POSES
│   ├── ACCAD
│   │   ├── Female1General_c3d.npz
│   │   ├── ...
│   │   ├── s011.npz
│   ├── BioMotionLab_NTroje
│   │   ├── rub001.npz
│   │   ├── ...
│   │   ├── rub115.npz
│   ├── ...
```

### 3. Preparing the pretrained FAISS model

For this step, you have to options: you can either download our , or train it from scratch by following the steps below

#### (Optional) 3.1 Download and extract the pretrained model

1. Download the [pretrained FAISS model](https://nc.mlcloud.uni-tuebingen.de/index.php/s/cxDEsezSXtDJfKt) (pretrained_models/FAISS_MODEL)
2. Once downloaded, extract it into `DATA_ROOT`

#### (Optional) 3.2 Train from scratch

Run:

```bash
python lib/data/prepare_faiss.py -d <DATA_ROOT>
```

#### Folders structure after this step:

```bash
<DATA_ROOT>
├── RAW_DATA
├── SAMPLED_POSES
├── FAISS_MODEL
│   ├── faiss.index
│   ├── all_data.npz
```

### 4. Noisy data and distance preparation

By running the commands below, you can efficiently prepare noisy data and distances either in a single execution or in parallel, depending on your requirements.

#### Single execution

To prepare noisy data and calculate distances in one step, run:

```bash
python lib/data/gen_data.py -c configs/data_gen.yaml
```

#### Parallel execution on SLURM

For parallel preparation of data, run:

```bash
python lib/data/gen_data_parallel.py -d <DATA_ROOT> -bf lib/data/gen_data.sh -sl
bash lib/data/gen_data.sh
```

**Notes:**

  * `-b` indicate the output bash file for data generation

  * Set `-sl` if you are using SLURM-based computing system

**Folders structure of traning data:**
```bash
<DATA_ROOT>
├── RAW_DATA
├── SAMPLED_POSES
├── FAISS_MODEL
├── NOISY_POSES
│   ├── gaussian_0.785
│   │   ├── ACCAD
│   │   │   ├── Female1General_c3d.npz
│   │   │   ├── ...
│   │   │   ├── s011.npz
│   │   ├── BioMotionLab_NTroje
│   │   │   ├── rub001.npz
│   │   │   ├── ...
│   │   │   ├── rub115.npz
│   │   ├── ...
```

