import os, sys, cv2, json, glob, math
import numpy as np
import pandas as pd
from scipy.stats import norm
import importlib.util
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Tuple

'''
extract and calculate scores for the Audio-corsi test
json_list: list [path_to_jsonfile, ...]
return:
    dict{
        uid:{
            "max_length": int; the max length of sequence the user could pass
            "correct_num": int; correct number of sequences user correctly recalled
            "overall_score": int; max_length * correct_num
            "weighted_score": float; a subtle score considering the length of the sequence
            "mm_score": normalized score (z-score) and scaled from 0 - 100
        }
    }
'''
def extract_corsi(json_list):
    corsi_results = {}
    corsi_scores = []
    # read every json file
    for json_path in json_list:
        filename = os.path.basename(json_path)
        _, _, uid, _, _, _ = filename.split("_")
        corsi_results[uid] = {}
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            max_length = 0
            correct_num = 0
            weight_score = 0
            # extract result for each trials
            for trial in data["trials"]:
                if trial["isCorrect"]:
                    correct_num += 1
                    weight_score += trial["spanLength"]
                    if trial["spanLength"] > max_length:
                        max_length = trial["spanLength"]
        # record the results
        corsi_results[uid]["max_length"] = max_length
        corsi_results[uid]["correct_num"] = correct_num
        corsi_results[uid]["overall_score"] = correct_num * max_length
        corsi_results[uid]["weighted_score"] = weight_score
        corsi_scores.append(correct_num * max_length)
    
    # normalizing overall_score using z-score
    # Scaling z-scores to a 0–100 range
    norm_score = 100 * norm.cdf((corsi_scores - np.mean(corsi_scores))/np.std(corsi_scores))
    idx = 0
    for uid, info in corsi_results.items():
        info["mm_score"] = round(float(norm_score[idx]), 2)
        idx += 1
    return corsi_results

'''
extract and calculate scores for the missing sound test
json_list: list [path_to_jsonfile, ...]
return:
    dict{
        uid:{
            "MAAE_deg": float;
            "LocationAccuracy_%": float;
            "CategoryAccuracy_%": float;
            "JointAccuracy_%": float;
        }
    }
'''
def extract_missing_result(json_list):
    missing_results = {}
    for json_path in json_list:
        filename = os.path.basename(json_path)
        _, _, uid, _, _, _ = filename.split("_")
        if uid not in missing_results:
            missing_results[uid] = {}
    
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract trials
        trials = data["trials"]
        df = pd.DataFrame(trials)
        # Compute metrics
        maae = df["angularErrorDeg"].mean()
        location_accuracy = df["locationCorrect"].mean() * 100
        sound_accuracy = df["soundCorrect"].mean() * 100
        joint_accuracy = ((df["locationCorrect"]) & (df["soundCorrect"])).mean() * 100
    
        missing_results[uid]["MAAE_deg"] = float(maae)
        missing_results[uid]["LocationAccuracy_%"] = float(location_accuracy)
        missing_results[uid]["CategoryAccuracy_%"] = float(sound_accuracy)
        missing_results[uid]["JointAccuracy_%"] = float(joint_accuracy)
    
    return missing_results


'''
extract and calculate scores for the background sound separation
json_list: list [path_to_rawfile, ...]
return:
    [
        [uid, N, E, S, W, NE, SE, SW, NW, Score],
        [uid, N, E, S, W, NE, SE, SW, NW, Score]
    ]
'''
def extract_gb_result(file_list):
    raw_data = pd.read_csv(file_list[0], skip_blank_lines=True)
    raw_data = raw_data.dropna(how='any')
    sentence_data = pd.read_csv(file_list[1])
    divisor = sentence_data.iloc[0]
    
    mapping = {
        'N': 'TOTAL_N',
        'E': 'TOTAL_E',
        'S': 'TOTAL_S',
        'W': 'TOTAL_W',
        'NE': 'TOTAL_NE',
        'SE': 'TOTAL_SE',
        'SW': 'TOTAL_SW',
        'NW': 'TOTAL_NW'
    }
    bg_result = raw_data.copy()
    for col, total_col in mapping.items():
        bg_result[col] = 1-raw_data[col] / divisor[total_col]
    
    cols = list(mapping.keys())
    bg_result['Score'] = bg_result[cols].mean(axis=1)
    return bg_result

def circular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute circular difference between two azimuths in degrees [0,180]."""
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def load_trials(json_path: Path) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Load trials and user info from JSON file.

    Returns:
        Tuple of (user_id, username, trials_list)
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Extract user info
    user_id = data.get("id", "")
    username = data.get("username", "")

    # Extract trials
    trials = data.get("trial", [])
    keep = []
    for t in trials:
        tgt = t.get("targetAz", None)
        rsp = t.get("responseAz", None)
        rt = t.get("reactionTime", None)
        if tgt is None or rsp is None or rt is None:
            continue
        keep.append(
            {
                "targetAz": float(tgt),
                "responseAz": float(rsp),
                "reactionTime": float(rt),
            }
        )
    return user_id, username, keep


def compute_metrics(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trials)
    correct_flags = [(t["targetAz"] % 360) == (t["responseAz"] % 360) for t in trials]
    n_correct = sum(1 for c in correct_flags if c)
    accuracy = n_correct / n if n > 0 else float("nan")
    # reaction time defined on correct trials (VASI convention)
    rts_correct = [t["reactionTime"] for t, c in zip(trials, correct_flags) if c]
    rt_mean_correct = mean(rts_correct) if rts_correct else float("nan")
    # also compute mean RT on all trials (fallback if no correct)
    rt_mean_all = mean([t["reactionTime"] for t in trials]) if n > 0 else float("nan")
    # Angular Error: circular absolute difference
    angular_errors = [
        abs(((t["responseAz"] - t["targetAz"]) + 180) % 360 - 180) for t in trials
    ]
    mean_angular_error = mean(angular_errors) if angular_errors else float("nan")
    return {
        "n_trials": n,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "rt_mean_correct": rt_mean_correct,
        "rt_mean_all": rt_mean_all,
        "angular_error_deg": mean_angular_error,
    }


def composite_score(
    rows: List[Dict[str, Any]], use_icc_weights: bool = True
) -> List[float]:
    """Compute 0–100 Spatial Ability Score from accuracy & RT (correct).
    Steps: z-score across batch -> weighted sum -> CDF -> 0–100.

    Weight strategy: Accuracy-dominant (70% accuracy, 30% reaction time)
    to ensure intuitive scoring where accuracy is the primary factor.
    """
    # Prepare arrays
    accs = [r["accuracy"] for r in rows]
    # Choose RT series: correct-only; fallback to all-trials when empty
    rts = [
        (
            r["rt_mean_correct"]
            if (r["rt_mean_correct"] == r["rt_mean_correct"])
            else r["rt_mean_all"]
        )
        for r in rows
    ]  # NaN check via x==x

    # No log transformation - use raw RT values
    # This prevents amplifying small differences in RT

    def zscore(vals: List[float]) -> List[float]:
        clean = [v for v in vals if v == v]
        if len(clean) <= 1:
            # If not enough data, return zeros
            return [0.0 for _ in vals]
        mu = sum(clean) / len(clean)
        sd = (sum((v - mu) ** 2 for v in clean) / (len(clean) - 1)) ** 0.5
        if sd == 0 or not (sd == sd):
            return [0.0 for _ in vals]
        return [(v - mu) / sd if (v == v) else 0.0 for v in vals]

    zA = zscore(accs)
    zRT_raw = zscore(rts)  # Use raw RT, not log-transformed
    # invert RT so faster -> higher
    zRT = [-z for z in zRT_raw]

    # Use accuracy-dominant weights: 70% accuracy, 30% reaction time
    # This makes scoring more intuitive - accuracy is the primary factor
    if use_icc_weights:
        wA, wRT = 0.7, 0.3  # Accuracy-dominant weighting
    else:
        wA, wRT = 0.5, 0.5  # Equal weights in equal-weights mode

    z_combined = [wA * a + wRT * r for a, r in zip(zA, zRT)]

    # Map z to 0–100 via normal CDF
    def phi(z: float) -> float:
        # standard normal CDF using erf
        return 0.5 * (1.0 + math.erf(z / (2.0**0.5)))

    scores = [round(phi(z) * 100.0, 2) for z in z_combined]
    return scores


def analyze_vasi_data(
    file_paths: List[str], use_icc_weights: bool = True
) -> List[Dict[str, Any]]:
    """Analyze VASI-VR JSON files and return results as a list of dictionaries.

    Args:
        file_paths: List of paths to JSON files (can be relative or absolute paths)
        use_icc_weights: If True, use accuracy-dominant weights (70/30),
                        otherwise use equal weights (50/50)

    Returns:
        List of dictionaries, each containing:
        - id: User ID
        - accuracy: Accuracy score (0-1)
        - reactionTime: Mean reaction time in seconds
        - angularError: Mean angular error in degrees
        - spatialAbilityScore: Composite spatial ability score (0-100)
    """
    # Convert string paths to Path objects and validate
    paths = []
    for fp in file_paths:
        p = Path(fp)
        # Add .json extension if not present
        if not p.suffix:
            p = Path(str(p) + ".json")
        if not p.exists():
            print(f"Warning: File not found: {p}")
            continue
        paths.append(p)

    if not paths:
        return []

    # Load and compute metrics for each file
    records = []
    for p in paths:
        user_id, username, trials = load_trials(p)
        m = compute_metrics(trials)
        records.append({"ID": user_id, "Username": username, **m})

    # Compute composite scores across the batch
    scores = composite_score(records, use_icc_weights=use_icc_weights)
    for rec, s in zip(records, scores):
        rec["SpatialAbilityScore(0-100)"] = s

    # Build output list with formatted values
    results = []
    for rec in records:
        results.append(
            {
                "id": rec["ID"],
                "accuracy": (
                    round(rec["accuracy"], 4)
                    if rec["accuracy"] == rec["accuracy"]
                    else None
                ),
                "reactionTime": (
                    round(rec["rt_mean_correct"], 4)
                    if rec["rt_mean_correct"] == rec["rt_mean_correct"]
                    else (
                        round(rec["rt_mean_all"], 4)
                        if rec["rt_mean_all"] == rec["rt_mean_all"]
                        else None
                    )
                ),
                "angularError": (
                    round(rec["angular_error_deg"], 2)
                    if rec["angular_error_deg"] == rec["angular_error_deg"]
                    else None
                ),
                "spatialAbilityScore": rec["SpatialAbilityScore(0-100)"],
            }
        )

    return results


def main():
    ## load all raw dataset
    audio_list = glob.glob(os.path.join("./dataset", "Audio*"))
    missing_list = glob.glob(os.path.join("./dataset", "Missi*"))
    vasi_list = glob.glob(os.path.join("./dataset", "VASI*"))
    bg_list = ["./dataset/raw_data.csv", "./dataset/sentence_data.csv"]
    
    ## extract indicator for all user study
    corsi_result = extract_corsi(audio_list)
    missing_result = extract_missing_result(missing_list)
    localtion_result = {}
    bg_result = extract_gb_result(bg_list)
    bg_result = {
        str(int(k)): v for k, v in bg_result.set_index("ID").to_dict(orient="index").items()
    }
    vasi_results = analyze_vasi_data(vasi_list, use_icc_weights=True)
    vasi_result = {}
    for x in vasi_results:
        uid = x["id"]
        x.pop("id", None)
        vasi_result[uid] = x

    final_result = {}

    # merge data
    keys = set(list(corsi_result.keys()) + list(missing_result.keys()) + list(vasi_result.keys()) + list(bg_result.keys()))

    for uid in keys:
        if uid not in final_result:
            final_result[uid] = {}
        final_result[uid].update(corsi_result.get(uid, {}))
        final_result[uid].update(missing_result.get(uid, {}))
        final_result[uid].update(vasi_result.get(uid, {}))
        final_result[uid].update(bg_result.get(uid, {}))
    # save integrated dataset into csv file
    df = pd.DataFrame.from_dict(final_result, orient='index')
    df.index.name = 'uid'
    df.reset_index(inplace=True)
    df.to_csv("preliminary_results.csv", index=False)


if __name__ == "__main__":
    main()