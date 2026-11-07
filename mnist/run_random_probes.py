"""Run the MNIST random probes for the two WiSARD rows in Table 2."""

import csv

import numpy as np

from pywisard.Wisard import BloomWisard, DictWisard
from pywisard.attacks.random.main import l2_noise, linf_noise
from pywisard.utils import get_mnist


def main():
    np.random.seed(0)  # Seed the noise in this recipe, not the tuple mappings.
    x_train, y_train, x_test, y_test = get_mnist()
    conditions = [
        ("clean", 0.0, x_test),
        ("random_linf", 0.1, np.array([linf_noise(x, 0.1) for x in x_test])),
        ("random_l2", 3.0, np.array([l2_noise(x, 3.0) for x in x_test])),
    ]
    settings = dict(
        n_tuples=48,
        n_nodes=50,
        bits_per_input=4,
        dtype=bool,
        withBleaching=False,
        n_jobs=1,
    )
    models = {
        "DictWiSARD": DictWisard(**settings),
        "Bloom WiSARD": BloomWisard(**settings, bloom_size=37412, num_hashes=4),
    }

    output = "mnist_random_probes_example.csv"
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "condition", "epsilon", "accuracy_pct", "retention_pct"])
        for name, model in models.items():
            model.train(x_train, y_train)
            accuracies = [
                float(np.mean(model.predict(images) == y_test))
                for _, _, images in conditions
            ]
            clean_accuracy = accuracies[0]
            for (condition, epsilon, _), accuracy in zip(
                conditions, accuracies, strict=True
            ):
                retention = f"{100 * accuracy / clean_accuracy:.4f}" if clean_accuracy else ""
                writer.writerow([name, condition, epsilon, f"{100 * accuracy:.4f}", retention])

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
