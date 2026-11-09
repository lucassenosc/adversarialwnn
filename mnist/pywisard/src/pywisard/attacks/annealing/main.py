"""Attack Success Rate of the Simulated Annealing (L2) attack on DictWisard.

Reproduces Table 8 of the paper: for each bits_per_input, trains a DictWisard
on MNIST and attacks every test image. An attack counts as a success when the
model's prediction on the final image differs from the true label. Images the
model already misclassifies are included (they count as successes).

Noise generators (--noise):
    threshold        one pixel crosses a thermometer threshold per step
                     (version used in the paper, with the budget-waste bug)
    threshold-fixed  same, but pixels returning to their original bin are
                     restored to their original value
    gaussian         dense Gaussian L2 step (create_l2_noise_generator)

Usage:
    python -m pywisard.attacks.annealing.main
    python -m pywisard.attacks.annealing.main --bits 1 2 --n-samples 100
"""

import argparse
import csv
import multiprocessing as mp
import os
import time

import numpy as np

from pywisard.Wisard import DictWisard
from pywisard.attacks.config import N_TUPLES, N_NODES, DTYPE, WITH_BLEACHING
from pywisard.l2attack import (
    create_l2_noise_generator,
    create_threshold_noise_generator,
    simulated_annealing,
)
from pywisard.utils import get_mnist

# Set before forking so worker processes share the trained model without pickling.
_MODEL = None
_ARGS = None


def attack_one(job):
    index, image, label = job
    # One seed per image, so results don't depend on the number of workers.
    np.random.seed(_ARGS.seed + index)

    image = image.reshape(1, -1)
    if _ARGS.noise == "gaussian":
        if _ARGS.norm != "l2":
            raise ValueError("The gaussian generator only supports --norm l2")
        add_noise = create_l2_noise_generator(image, _ARGS.epsilon)
    else:
        add_noise = create_threshold_noise_generator(
            image,
            _ARGS.epsilon,
            _MODEL.bits_per_input,
            restore=_ARGS.noise == "threshold-fixed",
            norm=_ARGS.norm,
        )
    # Early exit stops as soon as the image is misclassified. The random
    # trajectory up to that point is the same as without it, so the
    # success/failure outcome is identical — it just saves time.
    adv, _ = simulated_annealing(
        _MODEL,
        image,
        label,
        add_noise,
        step_size=_ARGS.step_size,
        step_decay=_ARGS.step_decay,
        n_iter=_ARGS.iters,
        with_early_exit=True,
        verbose=False,
    )
    pred = _MODEL.predict(adv)[0]
    return index, bool(pred != label), float(np.linalg.norm(adv - image))


def run(bits, X_train, y_train, X_test, y_test, args):
    global _MODEL

    print(f"\n=== bits_per_input = {bits} ===")
    # "auto": enough nodes to cover all 784 * bits input bits on average
    n_nodes = (
        -(-784 * bits // args.n_tuples) if args.n_nodes == "auto" else int(args.n_nodes)
    )
    print(
        f"n_tuples={args.n_tuples}, n_nodes={n_nodes}, "
        f"dtype={args.dtype}, bleaching={args.bleaching}"
    )
    model = DictWisard(
        n_tuples=args.n_tuples,
        n_nodes=n_nodes,
        bits_per_input=bits,
        dtype={"bool": bool, "int": int}[args.dtype],
        withBleaching=args.bleaching,
        n_jobs=1,
    )
    start = time.time()
    model.train(X_train, y_train)
    print(f"Training time: {time.time() - start:.1f}s")

    clean_acc = (model.predict(X_test) == y_test).mean()
    print(f"Clean accuracy: {clean_acc:.2%}")

    _MODEL = model
    jobs = [(i, X_test[i], y_test[i]) for i in range(len(X_test))]

    successes = 0
    start = time.time()
    with mp.get_context("fork").Pool(args.n_jobs) as pool:
        for done, (_, success, _) in enumerate(
            pool.imap_unordered(attack_one, jobs, chunksize=4), start=1
        ):
            successes += success
            if done % 500 == 0 or done == len(jobs):
                print(
                    f"[{done}/{len(jobs)}] ASR so far: {successes / done:.2%} "
                    f"({time.time() - start:.0f}s)",
                    flush=True,
                )

    asr = successes / len(jobs)
    print(f"Attack Success Rate: {asr:.2%}")
    return clean_acc, asr


def main():
    global _ARGS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument(
        "--noise",
        choices=["threshold", "threshold-fixed", "gaussian"],
        default="threshold",
    )
    parser.add_argument("--n-tuples", type=int, default=N_TUPLES)
    parser.add_argument(
        "--n-nodes", default=str(N_NODES), help='Integer, or "auto" to scale with bits'
    )
    parser.add_argument(
        "--dtype", choices=["bool", "int"], default=DTYPE.__name__
    )
    parser.add_argument(
        "--bleaching", action=argparse.BooleanOptionalAction, default=WITH_BLEACHING
    )
    parser.add_argument(
        "--norm",
        choices=["l2", "linf"],
        default="l2",
        help="Budget norm for the threshold generators",
    )
    parser.add_argument("--epsilon", type=float, default=1.58)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--step-decay", type=float, default=0.9999)
    parser.add_argument(
        "--n-samples", type=int, default=None, help="Attack only the first N test images"
    )
    parser.add_argument("--n-jobs", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="annealing_asr.csv")
    _ARGS = args = parser.parse_args()

    X_train, y_train, X_test, y_test = get_mnist()
    if args.n_samples is not None:
        X_test, y_test = X_test[: args.n_samples], y_test[: args.n_samples]

    print(
        f"noise={args.noise}, norm={args.norm}, epsilon={args.epsilon}, iters={args.iters}, "
        f"test images={len(X_test)}, workers={args.n_jobs}"
    )

    rows = []
    for bits in args.bits:
        clean_acc, asr = run(bits, X_train, y_train, X_test, y_test, args)
        rows.append((bits, clean_acc, asr))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bits_per_input", "clean_accuracy", "attack_success_rate"])
        writer.writerows(rows)

    print("\nbits_per_input, Attack Success Rate")
    for bits, _, asr in rows:
        print(f"{bits}, {asr:.2%}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
