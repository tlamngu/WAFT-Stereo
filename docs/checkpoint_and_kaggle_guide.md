# Cloud Checkpoint Strategy & Kaggle Integration Guide

This guide explains the upgraded checkpoint upload strategy, the automatic training recovery mechanism from the cloud, and how to utilize the Jupyter Notebook on Kaggle.

---

## ☁️ 1. Upgraded Checkpoint Upload Strategy

The checkpointing logic in [main.py](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/main.py) now operates under a highly optimized cloud upload strategy:

1. **Validation & Best Selection**:
   * During validation checks (every `cfg.TEST.EVAL_PERIOD` steps), the model compares the current average EPE (End-point Error) against the best EPE recorded so far (`best_epe`).
   * If a new lower EPE is found, it saves the weights locally to `checkpoint_best.pth` and marks a boolean flag `has_new_best = True`.

2. **Gated Uploads (Every 50 steps)**:
   * Every 50 training steps, the script checks if `has_new_best` is `True`.
   * If `True`, it uploads the local `checkpoint_best.pth` to Cloudflare R2 as `<dataset>_<config>_best.pth` and resets the flag.
   * If `False` (the best checkpoint did not change), it skips the upload completely. This prevents redundant uploads and saves network bandwidth.

3. **Latest Checkpoint Backup**:
   * Every `cfg.SOLVER.LATEST_CHECKPOINT_PERIOD` (usually 1,000 steps), the script saves the full optimizer states, step counts, and epoch counts to `checkpoint_latest.pth` and uploads it to Cloudflare R2 as `<dataset>_<config>_latest.pth`.

4. **Shutdown Safety Trigger**:
   * The training loop is wrapped in a `try...finally` block.
   * If the process is terminated (either successfully at `SOLVER.MAX_ITER`, interrupted via `Ctrl+C`, or killed by Kaggle timeouts), the `finally` clause automatically performs a final upload of the best model checkpoint before exiting.

---

## 🔄 2. Seamless Cloud Recovery (Resume Support)

To ensure training runs are recoverable across sessions or in case of notebook timeouts:
* At the start of `main.py`, the main process connects to the Cloudflare R2 bucket and searches for the unique cloud file key `<dataset>_<config>_latest.pth`.
* If found, it automatically downloads the file to your local checkpoint directory.
* The script then defrosts the configuration, sets `cfg.SOLVER.RESUME` to this local file path, and freezes it.
* The rest of the setup loads the model weights, optimizer states, last epoch, and last step count to continue training from where it left off.

---

## 🏆 3. Running on Kaggle

The notebook file [Kaggle_Training_Notebook.ipynb](file:///C:/Users/tlamn/Documents/AI/WAFT-Stereo/Kaggle_Training_Notebook.ipynb) is placed in the root directory. To run it:

1. Import the repository code into a Kaggle Notebook session.
2. In the Kaggle Notebook Editor, go to **Add-ons -> Secrets** and add your Cloudflare R2 credentials matching:
   * `CF_R2_ACCESS_KEY_ID`
   * `CF_R2_SECRET_ACCESS_KEY`
   * `CF_R2_ENDPOINT_URL`
   * `CF_R2_BUCKET_NAME`
3. If using Weights & Biases for monitoring, add `WANDB_API_KEY`.
4. Run the notebook cells sequentially:
   * **Step 1**: Sets up user credentials in `.env` and installs all library dependencies (`boto3`, `timm`, `peft`, `python-dotenv`).
   * **Step 2**: Performs a hardware preflight validation (CPU vs GPU capabilities).
   * **Step 3**: Verifies dataset layout and calibrator file presence for both `SanpoSynthetic` and `SanpoReal`.
   * **Step 4**: Validates the cloud recovery connection and prints a list of existing checkpoints in your R2 bucket.
   * **Step 5 (Dry-Run)**: Launches a 5-iteration dry-run on CPU. This compiles the MobileNetV4 model, initializes the data loading pipeline, passes batches, computes losses, updates weights, and saves checkpoints to verify correctness without using GPU resources or quotas.
   * **Step 6**: Starts the actual training loop on GPU (automatically resuming from cloud checkpoints if they exist).
   * **Step 7**: Evaluates the trained model.
