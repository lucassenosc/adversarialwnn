# MNIST experiments with PyWiSARD

Section 5 of *Binarization as an Attack Surface in Weightless Neural Networks* uses MNIST to study how image perturbations cross encoding thresholds and change predictions. This directory documents the available MNIST code. The [IDS guide](../../README.md) describes the separate network experiments in Section 6.

The [PyWiSARD snapshot](../../third_party/pywisard) provides DictWiSARD and Bloom WiSARD, with examples on MNIST. It comes from [theuvargas/pywisard at `ed76cd6`][revision] and retains the [original MIT license][license]. The script here runs the random-probe protocol for the two WiSARD models. See the [coverage table](#mnist-experiments-in-the-paper) for the files still needed to recreate the published results.

## Installation and data

Start in the root of the `adversarialwnn` checkout. Use Python 3.12 for MNIST and create a separate environment inside `experiments/mnist`.

Bash:

```bash
cd experiments/mnist
python3.12 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
cd experiments/mnist
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

In the activated environment, install the CPU wheels for Linux or Windows, then install PyWiSARD from the local snapshot. For macOS or CUDA, use the matching PyTorch 2.5.1 and torchvision 0.20.1 wheels in the [PyTorch installation instructions][torch].

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ../../third_party/pywisard
python -m pip check
```

Run the following commands from `experiments/mnist/`. The [MNIST loader][loader] downloads the standard 60,000 training and 10,000 test images through torchvision and caches them under `experiments/mnist/data/MNIST/`. Each image is scaled to `[0, 1]` and flattened to 784 features.

## Run the random probes

```bash
python run_random_probes.py
```

The script uses the model settings and perturbation budgets for the DictWiSARD and Bloom WiSARD rows of Table 2. Both models use a linear thermometer with 4 bits per input, 48 bits per tuple, and 50 RAMs per discriminator. Bloom WiSARD uses a Boolean filter of size 37,412 and 4 hashes. The script evaluates the clean test set and the same two perturbed test sets for both models.

The script writes `mnist_random_probes_example.csv` with accuracy and retention, where retention is 100 × perturbed accuracy ÷ clean accuracy. Treat the CSV as a fresh run. The script fixes the noise seed for this recipe. The library generates [tuple mappings][mapping] with a separate unseeded generator during training, so the accuracies can vary across runs.

## Public examples

These commands use the settings in the public code and train on the standard MNIST training set. The clean and random examples print accuracy to the terminal. The transfer example prints the transfer rate if it finds adversarial examples that fool its surrogate.

| Example | Command | Settings in the public code |
| --- | --- | --- |
| Clean classification | `python -m pywisard.attacks.clean.main` | DictWiSARD with 6 bits per input. [Source][clean] |
| Random probes | `python -m pywisard.attacks.random.main` | DictWiSARD and Bloom WiSARD with 1 bit per input, Random L∞ at ε = 0.1, and Random L2 at ε = 1.58. [Source][random] · [Settings][config] |
| Surrogate transfer | `python -m pywisard.attacks.transfer.main` | Bloom WiSARD with 1 bit per input, a 256/128 MLP trained on soft targets, and L2 MI-FGSM at ε = 3. [Source][transfer] · [Settings][config] |
| Exploratory Simulated Annealing | `python ../../third_party/pywisard/src/pywisard/l2attack.py` | One MNIST example against Bloom WiSARD using Gaussian proposals and an L2 budget of 2. It writes `original.png` and `adversary.png` in the current directory. [Source][legacy-sa] |

The transfer example uses four data-loader workers. In an environment that restricts communication between worker processes, set `num_workers=0` in the [training loader][trainer] to run the example with one process.

## MNIST experiments in the paper

| Paper result | What can be examined or run here | Files needed to recreate the published result |
| --- | --- | --- |
| **Model-agnostic probes** (Sections 4.1 and 5.2, Table 2, Figure 1) | The [noise functions][random] and the recipe above implement Random L∞ at ε = 0.1 and Random L2 at ε = 3 for DictWiSARD and Bloom WiSARD. Noise is clipped to the image range after perturbation. | The original run settings, tuple mappings or seeds, BTHOWeN evaluation, and the image index and noise used for Figure 1. |
| **BTHOWeN encoding comparison** (Section 5.3, Table 3, Figure 2) | PyWiSARD provides a [linear thermometer][linear] and Bloom WiSARD. The paper compares Gaussian and linear thermometer encodings within BTHOWeN while holding the other settings fixed. | The BTHOWeN revision and configuration used in the paper (6 bits per input, tuple size 49, 784 RAMs, counting filters of size 8,192 with 4 hashes, and bleaching). Include both encoders, the Gaussian threshold fitting code, and the 100 test-image indices used to count bit changes out of 4,704 bits per image under Random L∞ at ε = 0.1. |
| **Query-based Simulated Annealing** (Sections 4.2 and 5.4, Figure 3) | The [public SA example][legacy-sa] uses Gaussian proposals against Bloom WiSARD. The paper's attack queries DictWiSARD and proposes moving one pixel to a thermometer threshold at a time. It succeeds when the true-class score falls below the highest competing score. | The score-based threshold proposal and evaluation code for 1, 2, 4, and 6 bits per input. The reported settings are an L2 budget of 1.58, initial temperature 1.0, cooling rate 0.9, up to 5,000 iterations per image, and a temperature reset after five consecutive rejections. Include the evaluated sample indices and the trace or saved input and perturbation used for Figure 3. |
| **Surrogate transfer** (Sections 4.3 and 5.5, Table 5) | The [public transfer example][transfer] shows soft-target training and transfer to Bloom WiSARD. | The runs targeting DictWiSARD at 1, 2, 4, and 6 bits per input with a 256/256 MLP, soft WNN labels, and KL divergence. Include the four Foolbox 3.3.4 attacks (FGSM, BIM, MI-FGSM, and PGD) in L∞ at ε = 0.15, training and attack settings, and the three independent runs per setting. The aggregation should report both the surrogate fooling rate and the transfer rate among examples that fool the surrogate. |

For the transfer protocol in Section 4.3, FGSM uses one gradient-sign step. BIM and MI-FGSM use 10 steps of 0.03 with no random initialization. MI-FGSM uses momentum 1.0. PGD uses 40 steps of 0.005 and a random start within the budget. All inputs stay within `[0, 1]`.

A complete artifact for Section 5 should archive the original scripts and per-run configurations, the Python and package versions, the random seeds or saved mappings and model state, and the per-run outputs. It should include commands to regenerate Tables 2, 3, and 5, the results reported in Section 5.4, and Figures 1–3.

## License

PyWiSARD is released under the [MIT License][license].

[revision]: https://github.com/theuvargas/pywisard/tree/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1
[torch]: https://pytorch.org/get-started/previous-versions/
[loader]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/utils.py#L8-L33
[linear]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/utils.py#L36-L51
[mapping]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/Wisard.py#L118-L121
[clean]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/attacks/clean/main.py#L8-L48
[random]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/attacks/random/main.py
[config]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/attacks/config.py
[transfer]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/attacks/transfer/main.py
[trainer]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/attacks/transfer/trainer.py#L68-L80
[legacy-sa]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/src/pywisard/l2attack.py
[license]: https://github.com/theuvargas/pywisard/blob/ed76cd6f216a5ec8b9541ef0951f8018df2c3cd1/LICENSE
