#!/usr/bin/env python3
import re
from collections import defaultdict
import argparse
from pathlib import Path
import csv
import time


def create_default_stats():
    """Factory function to create default statistics dictionary."""
    return {"safe": 0, "unsafe": 0, "undecided": 0, "timeouts": 0}


def defaultdict_to_dict(d):
    """Convert nested defaultdict to regular dict for better printing."""
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def parse_log_result_dir(log_file_path, benchmark, verifier_name, csv_name=None, **kwargs):
    """Parse Venus log file to extract statistics by perturbation type and strength."""
    # Dictionary to store results: {data_name: {perturbation_type: {perturb_ratio: {robust_interval: {'safe': 0, 'unsafe': 0, 'undecided': 0, 'timeouts': 0}}}}}
    results = defaultdict(((lambda: defaultdict(create_default_stats))))

    # List to store CSV data
    csv_data = []

    for result_file in Path(log_file_path).rglob("*.txt"):
        print(result_file)
        is_sat, is_unsat, is_timeout = False, False, False
        runtime = None

        with open(result_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # Extract runtime if available
            runtime_match = re.search(r"(sat|unsat|timeout),([0-9]*\.?[0-9]+)", content, re.IGNORECASE)
            if runtime_match:
                runtime = float(runtime_match.group(2))

            if verifier_name == "venus":
                if "Result: unsafe" in content:
                    is_sat = True
                elif "Result: safe" in content:
                    is_unsat = True
                elif "Result: timeout" in content or "Result: unverified" in content:
                    is_timeout = True
                else:
                    is_timeout = True
            elif verifier_name == "neuralsat":
                if "unsat" in content.strip():
                    is_unsat = True
                elif "sat" in content.strip():
                    is_sat = True
                elif "timeout" in content.strip() or "early_stop" in content.strip() or "unknown" in content.strip():
                    is_timeout = True
                else:
                    print(content, result_file)
                    is_timeout = True
            elif verifier_name == "crown":
                if "unsat" in content.strip():
                    is_unsat = True
                elif "sat" in content.strip():
                    is_sat = True
                elif "timeout" in content.strip():
                    is_timeout = True
                else:
                    is_timeout = True
                    continue
                    raise Exception("Unknow result type of Crown")
            else:
                raise Exception("Unknown verifer name")

        # Determine status
        if is_sat:
            status = "unsafe"
        elif is_unsat:
            status = "safe"
        elif is_timeout:
            status = "timeout"
        else:
            status = "undecided"

        # Parse file path to extract components
        parts = result_file.parts
        if benchmark == "veridou":
            # Path format: benchmark_type/task/perturbation_type/kernel_size/strength/log_id.txt
            # Example: iclr_result/mnist_fc/motion_blur_30/5/0.2/log_9.txt
            if len(parts) >= 6:
                benchmark_type = parts[-6] if len(parts) > 6 else "unknown"
                task = parts[-7]
                kernel_size = parts[-5]
                perturbation_type = parts[-4]
                # kernel_size = parts[-3]
                C = parts[-3]
                strength = parts[-2]
                file_idx = result_file.parts[-1].split("_")[1].split(".")[0]
                with open(f"benchmarks/{task}/instances.csv", "r", encoding="utf-8", errors="ignore") as f:
                    instances = f.readlines()
                    model_name = instances[int(file_idx)].split(",")[0]
                    vnnlib_path = instances[int(file_idx)].split(",")[1]
                csv_data.append(
                    [
                        "image",
                        task,
                        model_name,
                        vnnlib_path,
                        kernel_size,
                        perturbation_type,
                        C,
                        strength,
                        status,
                        runtime,
                    ]
                )

                if is_sat:
                    results[strength][perturbation_type]["unsafe"] += 1
                elif is_unsat:
                    results[strength][perturbation_type]["safe"] += 1
                elif is_timeout:
                    results[strength][perturbation_type]["timeouts"] += 1
                else:
                    raise Exception("Undecided result")
        elif benchmark == "independent":
            # Path format: benchmark_type/task/perturbation_type/kernel_size/strength/log_id.txt
            # Example: iclr_result/mnist_fc/motion_blur_30/5/0.2/log_9.txt
            if len(parts) >= 6:
                benchmark_type = parts[-6] if len(parts) > 6 else "unknown"
                task = parts[-5]
                perturbation_type = parts[-4]
                kernel_size = parts[-3]
                strength = parts[-2]
                file_idx = result_file.parts[-1].split("_")[1].split(".")[0]
                with open(f"benchmarks/{task}/instances.csv", "r", encoding="utf-8", errors="ignore") as f:
                    instances = f.readlines()
                    model_name = instances[int(file_idx)].split(",")[0]
                    vnnlib_path = instances[int(file_idx)].split(",")[1]
                csv_data.append(
                    [
                        "image",
                        task,
                        model_name,
                        vnnlib_path,
                        kernel_size,
                        perturbation_type,
                        "-",
                        strength,
                        status,
                        runtime,
                    ]
                )

                if is_sat:
                    results[strength][perturbation_type]["unsafe"] += 1
                elif is_unsat:
                    results[strength][perturbation_type]["safe"] += 1
                elif is_timeout:
                    results[strength][perturbation_type]["timeouts"] += 1
                else:
                    raise Exception("Undecided result")
        else:
            raise Exception("Unknown benchmark type")

    # Save CSV data
    if csv_data:
        csv_filename = f"{benchmark}_results.csv" if csv_name is None else csv_name
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "benchmark_type",
                    "task",
                    "model_name",
                    "vnnlib_path",
                    "kernel_size",
                    "perturbation_type",
                    "perturb_ratio",
                    "strength",
                    "status",
                    "runtime",
                ]
            )
            writer.writerows(csv_data)
        print(f"CSV data saved to {csv_filename}")

    return results


def print_statistics(results, benchmark):
    """Print formatted statistics."""
    print("\n" + "=" * 80)
    print("VENUS VERIFICATION STATISTICS")
    print("=" * 80)

    # Convert defaultdict to regular dict for better printing
    results_dict = defaultdict_to_dict(results)
    print("Results structure:")
    print(results_dict)

    total_safe = 0
    total_unsafe = 0
    total_undecided = 0
    total_timeouts = 0

    for strength in sorted(results.keys()):
        print(f"\nStrength: {strength}" if benchmark == "independent" else f"\nPerturbation Ratio: {strength}")
        print("-" * 50)

        if benchmark == "veridou":
            for perturbation_type in sorted(results[strength].keys()):
                print(f"\n  Perturbation Type: {perturbation_type}")
                print("  " + "-" * 40)

                type_safe = 0
                type_unsafe = 0
                type_undecided = 0
                type_timeouts = 0

                stats = results[strength][perturbation_type]
                safe = stats["safe"]
                unsafe = stats["unsafe"]
                undecided = stats["undecided"]
                timeouts = stats["timeouts"]
                total = safe + unsafe + undecided + timeouts

                type_safe += safe
                type_unsafe += unsafe
                type_undecided += undecided
                type_timeouts += timeouts

                type_total = type_safe + type_unsafe + type_undecided + type_timeouts
                print(f"\n    Type Total:")
                print(
                    f"      SAFE={type_safe}, UNSAFE={type_unsafe}, UNDECIDED={type_undecided}, TIMEOUTS={type_timeouts}"
                )
                print(f"      Total={type_total}")

                total_safe += type_safe
                total_unsafe += type_unsafe
                total_undecided += type_undecided
                total_timeouts += type_timeouts
        else:
            data_safe = 0
            data_unsafe = 0
            data_timeouts = 0

            for pertubation_type in sorted(results[strength].keys(), key=str):
                stats = results[strength][pertubation_type]
                safe = stats["safe"]
                unsafe = stats["unsafe"]
                timeouts = stats["timeouts"]
                total = safe + unsafe + timeouts

                data_safe += safe
                data_unsafe += unsafe
                data_timeouts += timeouts

                print(f"\n    Perturbation Type {pertubation_type}:")
                print(f"      SAFE={safe}, UNSAFE={unsafe}, TIMEOUTS={timeouts}")
                print(f"      Total={total}")

            data_total = data_safe + data_unsafe + data_timeouts
            print(f"\n  Data Total:")
            print(f"    SAFE={data_safe}, UNSAFE={data_unsafe}, TIMEOUTS={data_timeouts}")
            print(f"    Total={data_total}")

            total_safe += data_safe
            total_unsafe += data_unsafe
            total_timeouts += data_timeouts

    print("\n" + "=" * 80)
    grand_total = total_safe + total_unsafe + total_undecided + total_timeouts
    print("OVERALL TOTALS:")
    print(f"  SAFE={total_safe}, UNSAFE={total_unsafe}, UNDECIDED={total_undecided}, TIMEOUTS={total_timeouts}")
    print(f"  Total Cases={grand_total}")
    if grand_total > 0:
        print(f"\nPERCENTAGES:")
        print(f"  SAFE: {(total_safe/grand_total)*100:.1f}%")
        print(f"  UNSAFE: {(total_unsafe/grand_total)*100:.1f}%")
        print(f"  UNDECIDED: {(total_undecided/grand_total)*100:.1f}%")
        print(f"  TIMEOUTS: {(total_timeouts/grand_total)*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--benchmark", choices=["independent", "VeriDou"], required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--csv_name", required=False)
    args = parser.parse_args()
    results = parse_log_result_dir(args.result_dir, args.benchmark, args.verifier, args.csv_name)
    print_statistics(results, args.benchmark)
