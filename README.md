
# Breast Cancer Detection Using Machine Learning and Deep Learning

This is a **near-exact rebuild package** of your original project based on the recovered source files.
It preserves your original code modules, notebook, algorithms, configuration, and expected project layout.

## What is included
- Original recovered Python files:
  - `train.py`
  - `predict.py`
  - `config.py`
  - `data_loader.py`
  - `enhancement.py`
  - `segmentation.py`
  - `model.py`
  - `visualization.py`
  - `breast_cancer_detection.ipynb`
- A `utils.py` compatibility file so the original imports work
- Pre-created folders for datasets, trained models, and outputs
- Helper scripts for setup and training

## What is NOT included
The following were **not present in the uploaded files**, so they are not bundled here:
- Original raw datasets
- Trained `.keras` model files
- Large output images generated during training

## Expected dataset folders
Place your datasets here exactly:

```text
/data/cbis_ddsm/
/data/histopathology/
```

## Expected trained model files
After training, these files should exist in `/models/`:
- `cnn_histopathology.keras`
- `cnn_watershed.keras`
- `cnn_canny.keras`

## Quick start

### 1. Create environment
```bash
pip install -r requirements.txt
```

### 2. Put datasets in the expected folders
- CBIS-DDSM → `data/cbis_ddsm/`
- Breast Histopathology Images → `data/histopathology/`

### 3. Train models
```bash
python train.py --mode all
```

### 4. Run prediction
```bash
python predict.py --image path/to/image.png --type histo
python predict.py --image path/to/image.png --type mammogram
```

### 5. Demo mode without datasets
```bash
python train.py --mode demo
python predict.py --demo --type histo
python predict.py --demo --type mammogram
```

## Files added in this rebuild
- `utils.py` — compatibility re-export file for your original imports
- `scripts/setup_folders.py` — recreates folder layout if needed
- `scripts/train_all.bat` — Windows helper script
- `scripts/run_demo.bat` — Windows helper script
- `docs/dataset_setup.md` — where to place datasets and expected files
- `docs/original_readme_snapshot.md` — preserved original README text

## Original README snapshot
Your original README content has been preserved in:
`docs/original_readme_snapshot.md`
