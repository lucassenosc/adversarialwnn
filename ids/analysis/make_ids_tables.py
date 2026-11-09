"""Regenerate the IDS tables of the paper from results/aggregated_reports/.

Protocol (Section 6.1): linear and Gaussian thermometer encodings; perturbations
FGSM, Random L2, and Random Linf (Table 6). The distributive encoding and the
C&W / BPDA / HSJA runs present in the CSVs are excluded.

Usage (from the repository root, or from ids/):
    python ids/analysis/make_ids_tables.py
Writes ids/results/aggregated_reports/ids_tables.tex
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "aggregated_reports"

ENCODINGS = ["linear", "gaussian"]
ATTACKS = ["FGSM", "RANDOM_L2", "RANDOM_LINF"]
ATTACK_TEX = {"FGSM": "FGSM", "RANDOM_L2": r"Random $L_2$", "RANDOM_LINF": r"Random $L_\infty$"}
MODELS = {"Standard": "standart_wisard", "Bloom": "bloom_wisard"}
DATASETS = ["Bot-IoT", "CICIDS", "UNSW-NB15"]
TASKS = ["binary", "multiclass"]
CLASS_TEX = {
    "Attack": "Malicious", "web attacks": "Web attacks", "bots": "Bots",
    "brute force": "Brute force", "port scanning": "Port scanning",
    "ddos": "DDoS", "dos": "DoS",
}


def load(model_prefix):
    frames = []
    for ds in DATASETS:
        for task in TASKS:
            path = AGG / f"{model_prefix}_{ds}_{task}_COMPLETO.csv"
            df = pd.read_csv(path, sep=";", decimal=",")
            df["Task"] = task
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df[df.Binarization.isin(ENCODINGS) & df.Attack.isin(ATTACKS)]


def table_by_attack(data):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"\textbf{Model} & \textbf{Perturbation} & \shortstack{\textbf{Acc. drop}\\\textbf{(pp)}}"
             r" & \textbf{MCC} & \textbf{ASR} & \textbf{FAR} \\", r"\midrule"]
    for i, (name, df) in enumerate(data.items()):
        if i:
            lines.append(r"\midrule")
        for atk in ATTACKS:
            x = df[df.Attack == atk]
            lines.append(f"{name} & {ATTACK_TEX[atk]} & {x.Acc_Drop_pp.mean():.2f} & "
                         f"{x.Adv_MCC.mean():.4f} & {x.Adv_ASR.mean():.4f} & {x.Adv_FAR.mean():.4f} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_by_dataset(data):
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"\textbf{Model} & \textbf{Dataset} & \textbf{Task} & \textbf{Acc.} & \textbf{Drop}"
             r" & \textbf{MCC} & \textbf{ASR} \\", r"\midrule"]
    for i, (name, df) in enumerate(data.items()):
        if i:
            lines.append(r"\midrule")
        for ds in DATASETS:
            for task in TASKS:
                x = df[(df.Dataset == ds) & (df.Task == task)]
                lines.append(f"{name} & {ds} & {task.capitalize()} & {x.Adv_Acc.mean():.4f} & "
                             f"{x.Acc_Drop_pp.mean():.2f} & {x.Adv_MCC.mean():.4f} & {x.Adv_ASR.mean():.4f} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def table_miss_by_class(data):
    std, blm = data["Standard"], data["Bloom"]
    lines = [r"\begin{tabular}{lllrr}", r"\toprule",
             r"\textbf{Dataset} & \textbf{Task} & \textbf{Class} & \textbf{Standard} & \textbf{Bloom} \\",
             r"\midrule"]
    for i, ds in enumerate(["CICIDS", "UNSW-NB15", "Bot-IoT"]):
        if i:
            lines.append(r"\midrule")
        rows = []
        for task in TASKS:
            s = std[(std.Dataset == ds) & (std.Task == task)]
            b = blm[(blm.Dataset == ds) & (blm.Task == task)]
            cols = [c for c in s.columns if c.startswith("Adv_Miss_") and s[c].notna().any()]
            rows += [(s[c].mean(), task, c[len("Adv_Miss_"):], b[c].mean()) for c in cols]
        for s_val, task, cls, b_val in sorted(rows, key=lambda r: -r[0]):
            label = CLASS_TEX.get(cls, cls.capitalize())
            lines.append(f"{ds} & {task.capitalize()} & {label} & {s_val:.2f} & {b_val:.2f} \\\\")
    return lines + [r"\bottomrule", r"\end{tabular}"]


def main():
    data = {name: load(prefix) for name, prefix in MODELS.items()}
    for name, df in data.items():
        print(f"{name}: {len(df)} runs after filtering")
    out = ["% Mean metrics by perturbation method", *table_by_attack(data), "",
           "% Mean metrics by dataset and task", *table_by_dataset(data), "",
           "% Mean miss rate by intrusion class", *table_miss_by_class(data), ""]
    path = AGG / "ids_tables.tex"
    path.write_text("\n".join(out))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
