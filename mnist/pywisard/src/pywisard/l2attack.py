import numpy as np
from pywisard.Wisard import Wisard, BloomWisard, DictWisard
from pywisard.utils import get_mnist
from pywisard.BloomFilter import bloom_filter_parameters
import time
import matplotlib.pyplot as plt


def main():
    X_train, y_train, X_test, y_test = get_mnist()

    size, hashes = bloom_filter_parameters(6000, 0.05)
    bloom_wisard = BloomWisard(
        n_tuples=48,
        n_nodes=150,
        bits_per_input=6,
        bloom_size=size,
        num_hashes=hashes,
        dtype=bool,
        withBleaching=False,
        n_jobs=1,
    )
    bloom_wisard2 = BloomWisard(
        n_tuples=48,
        n_nodes=50,
        bits_per_input=4,
        bloom_size=size,
        num_hashes=hashes,
        dtype=bool,
        withBleaching=False,
        n_jobs=1,
    )
    dict_wisard = DictWisard(n_tuples=48, n_nodes=50, bits_per_input=4, n_jobs=1)

    print("Training...")
    start = time.time()
    bloom_wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    i = 0
    adversary = generate_adversarial_example(
        bloom_wisard,
        X_test[i].reshape(1, -1),
        y_test[i],
        epsilon=2,
    )

    print(np.linalg.norm(X_test[i] - adversary))

    plot_number(X_test[i], "original")
    plot_number(adversary, "adversary")

    print("Training...")
    start = time.time()
    bloom_wisard2.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")

    print(
        f"Bloom Wisard 2 original prediction: {bloom_wisard2.predict_proba(X_test[i].reshape(1, -1))}"
    )
    print(
        f"Bloom Wisard 2 adversarial prediction: {bloom_wisard2.predict_proba(adversary.reshape(1, -1))}"
    )

    print("Training...")
    start = time.time()
    dict_wisard.train(X_train, y_train)
    print(f"Training time: {time.time() - start}")
    print(
        f"Other Wisard original prediction: {dict_wisard.predict_proba(X_test[i].reshape(1, -1))}"
    )
    print(
        f"Other Wisard adversarial prediction: {dict_wisard.predict_proba(adversary.reshape(1, -1))}"
    )


def plot_number(image, name):
    plt.imshow(image.reshape(28, 28), cmap="gray")
    plt.savefig(f"{name}.png")
    plt.close()


def generate_adversarial_example(
    model,
    image: np.ndarray,
    label,
    epsilon=0.1,
    step_size=1,
    step_decay=0.9999,
    target=None,
):
    original_image = np.copy(image)
    add_noise = create_l2_noise_generator(original_image, epsilon)
    adv_image, _ = simulated_annealing(
        model,
        original_image,
        label,
        add_noise,
        target=target,
        step_size=step_size,
        step_decay=step_decay,
    )

    print("============================================")
    print("Original prediction:", model.predict_proba(original_image))
    print("Adversarial prediction:", model.predict_proba(adv_image))
    return adv_image


def fitness(model: Wisard, image, label, target=None):
    counts = model.predict_proba(image)[0]
    total = sum(counts.values())

    if total == 0:
        return 0

    proba = {cls: cnt / total for cls, cnt in counts.items()}
    original_prob = proba.get(label, 0)

    if target is not None:
        other_probs = [prob for cls, prob in proba.items() if cls != target]
        max_other_prob = max(other_probs)
        target_prob = proba.get(target, 0)
        return target_prob - max_other_prob
    else:
        other_probs = [prob for cls, prob in proba.items() if cls != label]
        max_other_prob = max(other_probs)
        if not other_probs:
            return 0
        return max_other_prob - original_prob


def simulated_annealing(
    model: Wisard,
    image,
    label,
    add_noise,
    step_size,
    step_decay,
    target=None,
    n_iter=20_000,
    with_early_exit=False,
    verbose=True,
):
    current_image = np.copy(image)
    current_fitness = fitness(model, current_image, label, target)

    best_image = np.copy(current_image)
    best_fitness = current_fitness

    temperature = 10.0
    cooling_rate = 0.999
    min_temp = 0.001

    for i in range(n_iter):
        if with_early_exit and best_fitness > 0:
            break

        candidate_image = add_noise(current_image, step_size)
        step_size = max(step_size * step_decay, 0.1)

        candidate_fitness = fitness(model, candidate_image, label, target)

        delta = candidate_fitness - current_fitness

        max_exp = 20
        raw_exp = delta / temperature
        clipped_exp = np.clip(raw_exp, -max_exp, max_exp)
        probability = np.exp(clipped_exp) if temperature > 1e-10 else 0

        if delta >= 0 or np.random.random() < probability:
            current_image = candidate_image
            current_fitness = candidate_fitness

            if current_fitness > best_fitness:
                best_image = np.copy(current_image)
                best_fitness = current_fitness

        temperature = max(temperature * cooling_rate, min_temp)

        if verbose and i % 100 == 0:
            print(
                f"[{i}] {best_fitness:.4f} ({temperature:.4f}deg), step_size: {step_size:.4f}, Prob: {probability:.4f}",
            )

    return best_image, best_fitness


def create_l2_noise_generator(original_image, epsilon):
    def add_noise(current_image, step_size):
        delta = current_image - original_image

        perturbation = np.random.randn(*delta.shape)
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > 0:
            perturbation = (perturbation / perturbation_norm) * step_size
        else:
            perturbation = np.zeros_like(perturbation)

        new_delta = delta + perturbation
        new_delta_norm = np.linalg.norm(new_delta)
        if new_delta_norm > epsilon:
            new_delta = new_delta * (epsilon / new_delta_norm)

        candidate_image = np.clip(original_image + new_delta, 0, 1)
        return candidate_image

    return add_noise


def create_threshold_noise_generator(
    original_image,
    epsilon,
    bits_per_input,
    restore=False,
    norm="l2",
    margin=0.001,
    max_tries=100,
):
    """Moves one random pixel across one of its two neighbouring thermometer
    thresholds (50% chance each; pixels at the edges only have one option).

    Candidates that would exceed the L2 budget are rejected and redrawn. If no
    valid move is found after max_tries, the current image is returned.

    restore=False is the version used in the paper: a pixel that crosses back
    into its original bin stays just past the threshold, so it keeps spending
    budget without changing what the model sees.
    restore=True fixes this: a pixel that re-enters its original bin is set
    back to its original value, freeing the budget it used.

    norm is the budget: "l2" (epsilon bounds the L2 norm of the whole
    perturbation) or "linf" (every pixel may move at most epsilon).

    step_size is ignored; it is only accepted to match simulated_annealing.
    """
    if norm not in ("l2", "linf"):
        raise ValueError("norm must be 'l2' or 'linf'")
    thresholds = np.linspace(0, 1, bits_per_input + 1, endpoint=False)[1:]
    original = original_image.ravel()

    def bin_of(value):
        # Number of thresholds the value is at or above (same rule as binarize_dataset)
        return np.searchsorted(thresholds, value, side="right")

    def add_noise(current_image, step_size=None):
        current = current_image.ravel()

        for _ in range(max_tries):
            i = np.random.randint(current.size)
            b = bin_of(current[i])

            options = []
            if b > 0:  # cross the threshold below
                options.append(thresholds[b - 1] - margin)
            if b < len(thresholds):  # cross the threshold above
                options.append(thresholds[b] + margin)
            new_value = options[np.random.randint(len(options))]

            if restore and bin_of(new_value) == bin_of(original[i]):
                new_value = original[i]

            new_value = np.clip(new_value, 0, 1)
            if norm == "linf":
                # Only pixel i changes, so only its own distance matters
                if abs(new_value - original[i]) > epsilon:
                    continue
                candidate = current.copy()
                candidate[i] = new_value
                return candidate.reshape(current_image.shape)

            candidate = current.copy()
            candidate[i] = new_value
            if np.linalg.norm(candidate - original) <= epsilon:
                return candidate.reshape(current_image.shape)

        return current_image.copy()

    return add_noise


if __name__ == "__main__":
    main()
