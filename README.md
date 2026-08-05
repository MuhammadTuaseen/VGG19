# Retinal OCT Classification with VGG19

This project classifies retinal optical coherence tomography (OCT) scans into four diagnostic categories using transfer learning with **VGG19**. The complete workflow is provided in a Google Colab notebook.

> **Research use only.** This repository is an educational machine-learning project and is not a clinical diagnostic tool.

## Classes

The model is trained on the [Kermany 2018 OCT dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018), which contains scans from these classes:

- `CNV` — choroidal neovascularization
- `DME` — diabetic macular edema
- `DRUSEN`
- `NORMAL`

## Project contents

| File | Description |
| --- | --- |
| `Retinal_Optical_Coherence_Tomography_(OCT)usingVGG19.ipynb` | Colab notebook for downloading data, training, and evaluating the model. |

## Model

The notebook builds a classifier with:

- ImageNet-pretrained VGG19, without its original classification head
- Frozen VGG19 backbone
- Input size of `150 × 150 × 3`
- Two added convolutional layers (128 and 64 filters) with PReLU activations
- A dense layer with 100 units and a four-class softmax output
- Adam optimizer and categorical cross-entropy loss

Images are rescaled from `[0, 255]` to `[0, 1]`. Training is configured for 25 epochs with batches of 500 images.

## Requirements

Run the notebook in Google Colab with a GPU runtime if available. It uses:

```text
tensorflow
tensorflow-addons
numpy
matplotlib
seaborn
scikit-learn
kaggle
```

## Getting started

1. Create a Kaggle API token from your Kaggle account settings and download `kaggle.json`.
2. Open `Retinal_Optical_Coherence_Tomography_(OCT)usingVGG19.ipynb` in Google Colab.
3. Upload `kaggle.json` when prompted by the first cell.
4. Run the dataset download and extraction cells.
5. Update the dataset paths if your extraction directory differs from the notebook's `/content/eye-dataset/...` paths.
6. Run the remaining cells to train and evaluate the model.

The expected dataset structure is:

```text
OCT2017/
├── train/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
├── val/
│   ├── CNV/
│   ├── DME/
│   ├── DRUSEN/
│   └── NORMAL/
└── test/
    ├── CNV/
    ├── DME/
    ├── DRUSEN/
    └── NORMAL/
```

## Metrics

The notebook tracks accuracy, AUC, Cohen's kappa, F1 score, precision, and recall. Its saved output reports an evaluation accuracy of `0.9607` and AUC of `0.9972`.

### Reproducibility note

The saved per-class classification report in the notebook reports approximately 26% accuracy, which conflicts with the evaluation result above. Before relying on any metric, rerun evaluation with a non-shuffled test generator and reset it before prediction:

```python
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    class_mode="categorical",
    batch_size=50,
    shuffle=False,
)

test_generator.reset()
predictions = model_vgg.predict(test_generator)
predicted_classes = predictions.argmax(axis=1)
```

Then compare `predicted_classes` with `test_generator.classes`. This ensures predictions remain aligned with the true labels.

## Dataset citation

Kermany, D. S., Goldbaum, M., Cai, W., *et al.* (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. **Cell**, 172(5), 1122–1131.e9. https://doi.org/10.1016/j.cell.2018.02.010
