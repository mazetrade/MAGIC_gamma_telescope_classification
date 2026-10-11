# MAGIC Gamma Telescope Classification

A binary classification project that tells **gamma-ray events** apart from **hadron (cosmic-ray) background events** using simulated data from the MAGIC Cherenkov telescope. Four classic machine-learning models are trained and compared: **K-Nearest Neighbors, Naive Bayes, Logistic Regression and Support Vector Machine**.

---

## Table of contents

1. [Background: what is MAGIC?](#background-what-is-magic)
2. [The problem](#the-problem)
3. [Dataset](#dataset)
4. [Project workflow](#project-workflow)
5. [Models](#models)
6. [Results](#results)
7. [Key takeaways](#key-takeaways)
8. [Limitations and next steps](#limitations-and-next-steps)
9. [How to run](#how-to-run)
10. [Project structure](#project-structure)
11. [Acknowledgements](#acknowledgements)

---

## Background: what is MAGIC?

**MAGIC** (*Major Atmospheric Gamma Imaging Cherenkov* telescopes) is a pair of 17-metre telescopes at the Roque de los Muchachos Observatory on La Palma, Canary Islands. They study very-high-energy gamma rays from sources such as black holes, pulsars and supernova remnants.

Gamma rays never reach the ground. When one hits the upper atmosphere it triggers a cascade of particles called an **air shower**. These particles travel faster than light does in air, so they emit a brief flash of blue **Cherenkov light** that the telescope's camera records as an image, usually shaped like an ellipse.

## The problem

Cosmic rays made of **hadrons** (mostly protons) also create air showers, and they are far more common than gamma rays. Most of what the telescope sees is this background.

The two kinds of shower leave different footprints on the camera:

- **Gamma:** compact, regular ellipses that point back toward the source.
- **Hadron:** wider, irregular, more scattered images.

**Goal:** given the shape parameters of a recorded image, predict whether it was caused by a gamma ray (`1`) or a hadron (`0`).

## Dataset

- **Source:** [UCI Machine Learning Repository: MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)
- **File:** `gamma_telescope.data` (CSV with no header row)
- **Size:** 19,020 events, 10 numeric features and 1 label
- **Nature:** Monte Carlo **simulated** events, not real observations
- **Missing values:** none

### Features

The features are the *Hillas parameters*, a standard way of describing the ellipse that a shower leaves on the camera.

| Column | Description |
|---|---|
| `Length` | Major axis of the ellipse (mm) |
| `Width` | Minor axis of the ellipse (mm) |
| `Size` | log10 of the total light in the image (photo-electrons) |
| `Cone` | Ratio of the sum of the two brightest pixels to `Size` |
| `Concl` | Ratio of the brightest pixel to `Size` |
| `Asym` | Distance from the brightest pixel to the ellipse centre, projected on the major axis (mm) |
| `M3Long` | Third root of the third moment along the major axis (mm) |
| `M3Trans` | Third root of the third moment along the minor axis (mm) |
| `Alpha` | Angle between the major axis and the line to the camera centre (degrees) |
| `Dist` | Distance from the camera centre to the ellipse centre (mm) |
| `Class` | `g` = gamma (signal), `h` = hadron (background) |

### Class balance

| Class | Count | Share |
|---|---|---|
| Gamma (`g` → 1) | 12,332 | 64.8% |
| Hadron (`h` → 0) | 6,688 | 35.2% |

The dataset is **imbalanced**, which is handled during preprocessing (see below).

## Project workflow

All the work is in [`gamma_telescope.ipynb`](gamma_telescope.ipynb).

### 1. Loading and encoding

The raw file has no header, so column names are assigned when it is loaded with pandas. The text label is then converted to a binary integer:

```python
df["Class"] = (df["Class"] == "g").astype(int)   # gamma = 1, hadron = 0
```

### 2. Exploratory data analysis

For every feature, a normalised histogram compares the gamma distribution (blue) with the hadron distribution (red). This shows which features separate the two classes well. For example, `Alpha` is concentrated near small values for gamma events, which matches the physics: gamma showers point toward the source at the camera centre.

### 3. Train / validation / test split

The data is shuffled and split **60 / 20 / 20**:

```python
train, valid, test = np.split(df.sample(frac=1),
                              [int(0.6 * len(df)), int(0.8 * len(df))])
```

| Set | Rows |
|---|---|
| Train | 11,412 |
| Validation | 3,804 |
| Test | 3,804 |

### 4. Scaling and oversampling

A helper function `scale_dataset` prepares each subset:

1. **Standardisation** with `StandardScaler`, so every feature has mean 0 and standard deviation 1. Distance-based models such as KNN and SVM depend on this, because otherwise features with large ranges (`Dist`, `Length`) would dominate.
2. **Random oversampling** with `RandomOverSampler` from `imbalanced-learn`, applied to the **training set only**. Hadron events are duplicated at random until both classes have the same count, so the models don't simply learn to favour gamma.

After oversampling, the training set contains **14,776 rows: 7,388 per class**. The validation and test sets keep their natural imbalance so that evaluation reflects real conditions.

### 5. Training and evaluation

Each model is trained on the balanced training set and evaluated on the untouched test set with scikit-learn's `classification_report`, which gives precision, recall and F1-score for each class.

## Models

| Model | scikit-learn class | Settings | Idea |
|---|---|---|---|
| K-Nearest Neighbors | `KNeighborsClassifier` | `n_neighbors=3` | Labels a point by majority vote of its 3 closest training points |
| Naive Bayes | `GaussianNB` | default | Applies Bayes' theorem, assuming features are independent and normally distributed |
| Logistic Regression | `LogisticRegression` | default | Fits a linear decision boundary and outputs a probability through the sigmoid function |
| Support Vector Machine | `SVC` | default (RBF kernel) | Finds the boundary with the widest margin between classes; the RBF kernel allows non-linear boundaries |

## Results

Test set: **3,804 events** (1,351 hadron, 2,453 gamma).

| Model | Accuracy | Hadron precision | Hadron recall | Gamma precision | Gamma recall | Macro F1 |
|---|---|---|---|---|---|---|
| **SVM** | **0.86** | **0.81** | **0.80** | **0.89** | **0.90** | **0.85** |
| KNN (k=3) | 0.82 | 0.74 | 0.74 | 0.86 | 0.86 | 0.80 |
| Naive Bayes | 0.73 | 0.69 | 0.41 | 0.73 | 0.90 | 0.66 |
| Logistic Regression | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

> Results depend on the random shuffle in step 3, so re-running the notebook gives slightly different numbers.

## Key takeaways

- **SVM performs best**, with 86% accuracy and the most balanced performance across both classes. The RBF kernel captures the non-linear boundary between the two shower types.
- **KNN is a solid second** at 82%. It is simple, but it relies heavily on scaling and gets slower as the dataset grows.
- **Naive Bayes struggles with hadrons**: it finds only 41% of them. Its assumption that features are independent doesn't hold here, since `Length`, `Width` and `Size` are strongly correlated. It ends up labelling most events as gamma.
- **Accuracy alone is misleading** on imbalanced data. Recall on the minority class (hadron) shows the differences between models much more clearly.

## Limitations and next steps

- **Validation set not used yet.** It is the natural place to tune hyperparameters (for example `k` in KNN, `C` and `gamma` in SVM) before a final check on the test set.
- **Scaler fitted separately on each subset.** The scaler should be fitted on the training set only and then reused for validation and test, so that all three use the same transformation and no information leaks from test data.
- **No fixed random seed.** Adding `random_state` to `df.sample()` and `RandomOverSampler` would make results reproducible.
- **Possible improvements:**
  - Cross-validation and grid search for hyperparameter tuning
  - Tree-based models (Random Forest, Gradient Boosting, XGBoost)
  - A small neural network
  - ROC curves and AUC, which are commonly used for this dataset in physics, where the cost of misclassifying a hadron as gamma is high

## How to run

### Requirements

- Python 3.10+
- numpy, pandas, matplotlib, scikit-learn, imbalanced-learn, jupyter

### Setup

```bash
# create and activate a virtual environment
python -m venv env
env\Scripts\activate          # Windows
# source env/bin/activate     # macOS / Linux

# install dependencies
pip install numpy pandas matplotlib scikit-learn imbalanced-learn jupyter
```

> The package is installed as **`imbalanced-learn`** but imported as **`imblearn`**.

### Run

```bash
jupyter notebook gamma_telescope.ipynb
```

Then run all cells from top to bottom. Cells reuse variables from earlier cells, so after a kernel restart use **Run All** rather than running a single cell.

## Project structure

```
Gamma_telescope_classification/
├── gamma_telescope.data     # MAGIC Gamma Telescope dataset (UCI)
├── gamma_telescope.ipynb    # EDA, preprocessing, model training and evaluation
└── README.md
```

## Acknowledgements

- **Dataset:** Bock, R. (2004). *MAGIC Gamma Telescope* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52C8B
- **Telescope:** [MAGIC Collaboration](https://magic.mpp.mpg.de/)
- **Libraries:** scikit-learn, imbalanced-learn, pandas, NumPy, Matplotlib
