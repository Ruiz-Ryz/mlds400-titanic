# Titanic survival prediction

Rui Zhou — MLDS 400, Homework 3.

This repository reads the Kaggle Titanic files, prints a short exploration and
the preprocessing decisions, fits logistic regression, reports accuracy, and
writes predictions. The dataset is deliberately excluded from Git and Docker
images. Run every command below from the repository root unless stated otherwise.

## Requirements

- Git and a running Docker installation, such as Docker Desktop.
- A Kaggle account to download the competition files. No Kaggle token is needed
  inside either container.
- A terminal: Bash/Zsh on macOS/Linux, or PowerShell on Windows.

Check that Docker is running with `docker info`. The first build downloads the
base image and packages and therefore needs an internet connection.

Source layout:

```text
src/python/  main.py, requirements.txt, Dockerfile
src/r/       main.R, install_packages.R, Dockerfile
src/data/    train.csv and test.csv, created locally and ignored by Git
outputs/     generated metrics and predictions, ignored by Git
```

## 1. Clone and download the data

```bash
git clone https://github.com/Ruiz-Ryz/mlds400-titanic.git
cd mlds400-titanic
mkdir -p src/data outputs/python outputs/r
```

In PowerShell, replace the last command with:

```powershell
New-Item -ItemType Directory -Force src/data, outputs/python, outputs/r
```

1. Open the [official Titanic Data page](https://www.kaggle.com/competitions/titanic/data).
2. Sign in and join the competition/accept its rules if Kaggle prompts you.
3. Download and extract the competition ZIP, or download `train.csv` and
   `test.csv` individually.
4. Copy **only `train.csv` and `test.csv`** into `src/data/`. Keep the filenames
   unchanged. They should be `src/data/train.csv` and `src/data/test.csv`, not
   inside another nested `titanic/` folder.

The standard files contain 891 training rows and 418 test rows. `train.csv`
includes `Survived`; the official `test.csv` does not. The example
`gender_submission.csv` is a prediction baseline, **not true test labels** and
must not be used to calculate accuracy.

## 2. Build and run Python

The image uses Python 3.12. NumPy, pandas, and scikit-learn versions are pinned in
`src/python/requirements.txt`. Build from the **repository root**:

```bash
docker build -f src/python/Dockerfile -t mlds400-titanic-python .
```

Run on macOS/Linux:

```bash
docker run --rm \
  --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" \
  --mount "type=bind,source=$(pwd)/outputs/python,target=/app/output" \
  mlds400-titanic-python
```

Run on Windows PowerShell:

```powershell
docker run --rm --mount "type=bind,source=$($PWD.Path)/src/data,target=/app/data,readonly" --mount "type=bind,source=$($PWD.Path)/outputs/python,target=/app/output" mlds400-titanic-python
```

The input mount is read-only. The output mount saves files on your computer
before the temporary container is removed. A missing file or column produces a
clear terminal error rather than silently running on different data.

## 3. Build and run R

The R image uses `rocker/r-ver:4.4.3`. `install_packages.R` installs/verifies
`jsonlite` version 2.0.0 from CRAN. Reading the CSVs and fitting binomial logistic
regression use base R and `stats`, so no additional modeling packages are needed.

Build from the repository root:

```bash
docker build --platform linux/amd64 -f src/r/Dockerfile -t mlds400-titanic-r .
```

Run on macOS/Linux:

```bash
docker run --rm --platform linux/amd64 \
  --mount "type=bind,source=$(pwd)/src/data,target=/app/data,readonly" \
  --mount "type=bind,source=$(pwd)/outputs/r,target=/app/output" \
  mlds400-titanic-r
```

Run on Windows PowerShell:

```powershell
docker run --rm --platform linux/amd64 --mount "type=bind,source=$($PWD.Path)/src/data,target=/app/data,readonly" --mount "type=bind,source=$($PWD.Path)/outputs/r,target=/app/output" mlds400-titanic-r
```

The R image is explicitly built/run as `linux/amd64` to also support an Apple
Silicon Mac through Docker Desktop's emulation. The first R build can be slower.
Both languages use the same local input files, with separate output directories.

## Model and outputs

The model uses `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
It excludes names, tickets, cabin text, and passenger IDs from the predictors.
Numeric missing values are filled with training medians and scaled. Categorical
missing values are filled with training modes and one-hot encoded. No passenger
rows are dropped. All changes and fitted fill values are printed in the terminal.

First, a stratified 80/20 split with seed 42 gives a holdout validation accuracy.
Preprocessing is fitted on the 80% training portion only. Then a fresh model is
fitted to all labeled training rows, and its training accuracy is reported before
it predicts the competition test rows.

**Q18 limitation:** the original Kaggle `test.csv` has no `Survived` labels, so
true test accuracy cannot be calculated locally. The script prints `N/A` and
stores JSON `null` for that score. The holdout score is explicitly labeled
**validation accuracy**, not Kaggle test accuracy. If genuine test labels are
provided as a `Survived` column in `test.csv`, the script also evaluates them;
the label column is never used as an input feature.

Each script writes these two files into its own output directory
(`outputs/python/` or `outputs/r/`):

- `predictions.csv`: `PassengerId,Survived`, one binary prediction per test row.
- `metrics.json`: training/validation accuracy, unavailable test accuracy, and row counts.

Predictions can be uploaded to Kaggle manually for its competition evaluation.
Nothing is submitted to Kaggle automatically.

R uses factors to encode categorical variables and maps any new test categories
to the fitted training mode. Python one-hot encodes categories and ignores unseen
ones. Python uses scikit-learn's default L2-regularized logistic regression; R
uses unregularized `glm(..., family = binomial())`. Both use a 0.5 decision
threshold. The random split generators also differ between languages even with
the same seed, so their validation scores need not match and should not be used
as a controlled comparison of Python versus R.

## Optional Python run without Docker

With Python 3.12 installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r src/python/requirements.txt
python src/python/main.py --data-dir src/data --output-dir outputs/python
```

On Windows use `py -3.12 -m venv .venv`, followed by
`.venv\Scripts\Activate.ps1`. Docker is the primary reproducible environment.

For a native R installation (R 4.4 or newer recommended), run:

```bash
Rscript src/r/install_packages.R
Rscript src/r/main.R --data-dir src/data --output-dir outputs/r
```

Installing jsonlite from source requires a working compiler and write permission
to an R package library. The R Docker image supplies this environment.

## Git workflow

Development uses `main`, `develop`, and `feature/setup`, `feature/python`, and
`feature/r` branches. Feature work goes through pull requests into `develop`;
release pull requests merge `develop` into `main`.

To use GitHub Desktop instead of the terminal clone command, choose **File →
Clone Repository → URL**, enter this repository's URL, and choose a local path.
Open the cloned repository in your text editor using Desktop's **Repository →
Open in [your editor]** option. Continue with the same data and Docker steps above.

## Container verification

The **Docker checks** workflow in the Actions tab builds and runs both
images on an Ubuntu runner. It checks output columns, row/ID preservation,
binary predictions, metric ranges, and the missing Kaggle test score. It also
checks that the built images do not contain the input data.

For that check only, the runner downloads the Titanic files from the
[public dataset mirror maintained by Aurélien Géron](https://github.com/ageron/data/tree/main/titanic),
pinned to commit `0366cd1ba5d538e1fe807e7527d848f9585f9959`. The CSVs exist only
in the temporary runner workspace; they are not committed or uploaded as
artifacts. The normal user instructions above use the official Kaggle download.

It runs on pushes to `feature/r` and can also be run manually. To repeat the
check: open **Actions → Docker checks → Run workflow**, choose a
branch, and click **Run workflow**. The logs include the versions, preprocessing
decisions, accuracies, and output checks for both implementations.

## Troubleshooting

- **Cannot connect to Docker:** start Docker Desktop and wait until `docker info` succeeds.
- **Bind source path does not exist:** create the directories in step 1 and run from the repository root.
- **Missing train/test file:** extract the Kaggle ZIP and check the exact filenames in `src/data/`.
- **An empty test-accuracy value:** expected for Kaggle's unlabeled test file; see Q18 above.
- **R architecture warning on an ARM Mac:** keep `--platform linux/amd64` in both R commands.
- **Permission denied writing output on Linux:** add `--user "$(id -u):$(id -g)"` after `docker run`.

`.gitignore` excludes the dataset, predictions, environments, and session files.
`.dockerignore` also excludes the dataset from the build context. Data is only
made available to a running container through the read-only mount.
