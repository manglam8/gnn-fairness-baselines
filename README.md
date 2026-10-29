# Fairness and Bias in Graph Neural Networks: Baselines

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manglam8/gnn-fairness-baselines/blob/main/GNN_Fairness_midterm.ipynb)

Course project, **EECS 6412 - Data Mining, York University (Fall 2026)**. Midterm stage: implementation and evaluation of baseline methods.
Author: Kumar Manglam Pandey.

---

## 1. What is this project about?

A **graph** is a set of things connected to each other. In this project the things are people and the connections are
friendships or similarities. For example, in the Pokec social network each person is a **node**. A friendship between two
people is an **edge**.

A **Graph Neural Network (GNN)** is a machine-learning model that makes a prediction about each person (for example,
"works in field X: yes/no"). To do so it uses that person's own information **plus the information of their friends**.
This "look at your friends" step is called **message passing**.

**Why can this be unfair?** People tend to befriend people like themselves (sociologists call this *homophily*).
On Pokec, 95% of friendships are between people from the same region. So even if we hide a person's region from the
model, their friends "leak" it back in. The model's predictions can then end up depending on region, a
**sensitive attribute** (like gender, age or race) that should not influence the decision.

**What this repository does:**

1. **Measures** how much the graph leaks the sensitive attribute (diagnostics).
2. **Compares** a normal neural network (MLP, which ignores friendships) with a GNN (GCN, which uses them) to see what
   message passing does to accuracy and fairness.
3. **Re-implements and evaluates two published fairness methods** (the "baselines"):
   - **FairGNN** (Dai & Wang, WSDM 2021): trains the GNN to *fool a detective* that tries to guess the sensitive
     attribute.
   - **NIFTY** (Agarwal, Lakkaraju & Zitnik, UAI 2021): trains the GNN so that its output *does not change when the
     sensitive attribute is flipped* (for example, "would the prediction be the same if this person were from the other region?").
4. **Checks whether the published results can be reproduced**, and compares all methods fairly, under one common protocol,
   with statistical significance tests.

---

## 2. What is in this repository?

| File / folder | What it is |
|---|---|
| `GNN_Fairness_midterm.ipynb` | **The whole project in one notebook.** Open it in Google Colab and run it top to bottom. |
| `results/` | All result tables (`.csv`) and figures (`.png`) produced by the notebook, so you can look at the results **without running anything**. |
| `results/archive_seed19/` | Results from an earlier data split, kept as a robustness check (see Section 8). |
| `README.md` | This guide. |

**Datasets are not stored here.** They are large (about 325 MB) and belong to their original authors. The notebook
downloads them automatically (see Section 5).

---

## 3. How to run it (no installation needed)

You only need a Google account and a web browser.

1. **Click the "Open in Colab" badge** at the top of this page. The notebook opens in Google Colab, a free online Python
   environment.
2. **Turn on the free GPU** (it makes training about 10× faster):
   menu **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**.
3. **(Optional but recommended) Connect Google Drive** by running the first cell (Cell 0) and clicking "Allow".
   Results are then saved to `My Drive/gnn_fairness/`, so nothing is lost if Colab disconnects.
   If you skip this, results go to a temporary folder that disappears when Colab closes.
4. **Run all cells in order**: menu **Runtime → Run all**. Or click inside each cell and press **Shift + Enter**.
5. **Wait.** The full run takes about **2.5 hours** on a free T4 GPU (see the timing table below). You can close the
   tab while it runs. If it disconnects, just run everything again: finished experiments are **skipped automatically**
   (the notebook remembers what it already did).
6. **Read the results** in the outputs of Cells 14 and 15, or in the `results/` folder.

### How long does each part take?

| Cells | What happens | Time (T4 GPU) |
|---|---|---|
| 0–4 | Setup, download data, define functions | ~2 min |
| 5 | Train MLP and GCN models (100 runs) | ~30 min |
| 6–8 | Summary tables, significance tests | ~3 min |
| 9–12 | Define FairGNN and NIFTY, quick smoke test | ~1 min |
| 13 | Train all baselines (80 runs) | ~1.5–2 h |
| 14–15 | Reproduction table, final comparison, figure | ~3 min |

**Just want to check the code works?** Run Cells 0–4, then 9–12. Cell 12 is a one-minute "smoke test" that trains every
method briefly and prints `Smoke test passed`.

---

## 4. What does each part of the notebook do?

| Cell | Plain-English description | Output |
|---|---|---|
| 0 | Connects Google Drive (optional) | — |
| 1 | Downloads the datasets from the original authors' GitHub pages | — |
| 2 | Loads each dataset into four pieces: **X** (each person's features), **y** (what we predict), **s** (the sensitive attribute) and **A** (who is connected to whom). Splits people into training, validation and test sets **exactly as the original papers did**. | dataset sizes |
| 3 | **Diagnostics:** how often friends share the same sensitive group, which features act as hidden proxies for it, and how easily it can be guessed after "looking at friends" 0, 1 or 2 times | `m2_diagnostics.csv` |
| 4 | Defines the MLP/GCN model, the training loop and all fairness metrics | — |
| 5 | Trains MLP (1–2 layers) and GCN (1–3 layers) on 4 datasets × 5 random seeds | `m3_results_v4.csv` |
| 6 | Averages the results, makes a bar chart | `m3_summary_v4.csv`, `m3_mlp_vs_gcn_v4.png` |
| 7 | Statistical tests: is the GCN really fairer or less fair than the MLP, or is it noise? | `m3_paired_tests_v4.csv` |
| 8 | Shows how the choice of decision threshold changes fairness conclusions | `m3_posthoc_*_v4.csv` |
| 9 | Shared building blocks for the baselines | — |
| 10 | **FairGNN** (adversarial method) | — |
| 11 | **NIFTY** (counterfactual and stability method) | — |
| 12 | Smoke test (1 minute) | prints `Smoke test passed` |
| 13 | Trains 4 methods × 4 datasets × 5 seeds = 80 runs | `m4_results.csv` |
| 14 | **Reproduction check:** published numbers vs. ours | `m4_reproduction.csv` |
| 15 | **Final comparison** of all methods, significance tests, fairness–accuracy figure | `m4_main_table.csv`, `m4_paired_tests.csv`, `m4_tradeoff.png` |

---

## 5. Datasets

All four datasets are standard benchmarks in fair-graph-learning research. The notebook downloads them automatically from
the original authors' repositories:

- FairGNN repository: <https://github.com/EnyanDai/FairGNN> (version used: commit `13cdca7`), folder `dataset/pokec`
- NIFTY repository: <https://github.com/chirag126/nifty> (version used: commit `6c270c5`), folder `dataset`

| Dataset | People (nodes) | Connections (edges) | What is predicted | Sensitive attribute | Where the edges come from |
|---|---|---|---|---|---|
| **German Credit** | 1,000 | 21,742 | good / bad credit risk | gender | *Synthetic*: two people are linked if their bank profiles are similar |
| **Credit Defaulter** | 30,000 | 1,421,858 | will / won't default on payment | age group | *Synthetic*: similar spending/payment patterns |
| **Pokec-z** | 67,796 | 617,958 | working field (binarised) | region | *Real* friendships, Slovak social network |
| **Pokec-n** | 66,569 | 517,047 | working field (binarised) | region | *Real* friendships, a different set of provinces |

**Example.** In Pokec-z, a node might be a 24-year-old user who speaks English and German and lists sports and music as
hobbies. The task is to predict their field of work. The sensitive attribute is which of two regions they live in. Region
is **removed** from the features the model sees, yet the model can still recover it from the user's friends.

---

## 6. The methods in one paragraph each

**MLP** (multilayer perceptron): a standard neural network that looks only at each person's own features and ignores the
graph. It is our "no message passing" reference.

**GCN** (graph convolutional network, Kipf & Welling 2017): before each layer, every person's information is averaged
with their friends'. In our code the MLP and GCN are the same class and differ by **one line**: the averaging step.

**FairGNN.** Imagine a detective (the *adversary*) who looks at the GNN's internal summary of each person and tries to
guess their sensitive attribute. FairGNN trains the GNN to predict the label well **and** to make the detective fail. It
also penalises any correlation between predictions and the sensitive attribute. FairGNN only knows the true sensitive
attribute for a few people (200 on Pokec), so it first trains a small *estimator* to guess it for everyone else.

**NIFTY.** For each person NIFTY creates two slightly altered copies of the graph. Some edges are dropped and a little
noise is added to the features. In the second copy, **the person's sensitive attribute is flipped**. NIFTY then trains the
GNN so the two copies give the **same** internal summary. If flipping your group never changes your representation, the
prediction cannot depend on it.

**"GCN-FG" and "GCN-NF"** are each method's own GCN with the fairness parts switched off. They let us see exactly what the
fairness mechanism changes.

---

## 7. How fairness is measured

We split people into the two sensitive groups (for example, region 0 vs. region 1) and compare how the model treats them.

| Metric | Question it answers | Ideal |
|---|---|---|
| **ΔSP** (statistical parity gap) | Do both groups receive a "positive" prediction equally often? | 0 |
| **ΔEO** (equal opportunity gap) | Among people who truly *are* positive, are both groups recognised equally often? | 0 |
| **Excess SP** | Is the prediction gap larger than the gap that already exists in the true labels? | 0 |
| **Score leakage** | Can someone guess the group just from the model's score? (50% = no better than a coin flip) | 50 |
| **Counterfactual unfairness** (NIFTY) | What percentage of predictions change if we flip only the person's sensitive attribute? | 0 |
| **Instability** (NIFTY) | What percentage of predictions change if we add small random noise to the features? | 0 |

**Worked example.** Suppose the model approves 60% of region-0 users and 50% of region-1 users. Then ΔSP = 10 points.
If 55% of region-0 users and 50% of region-1 users are *truly* positive, the labels already differ by 5 points. So the
**excess** gap the model adds is 10 − 5 = **5 points**.

Gaps are reported **with a sign** (group 1 minus group 0), so you can see *who* is disadvantaged, not just how much.

---

## 8. Main results (5 seeds each; full tables in `results/`)

**Reproduction of published numbers** (each paper's own evaluation rules):

- NIFTY on Credit: **reproduced** (all six metrics within 0.7 points of the paper).
- Plain GCN on Pokec: **reproduced** accuracy (within 0.7 points). Our bias is slightly lower than published.
- FairGNN on Pokec: **partially reproduced**. It cuts the parity gap by about two-thirds, but not to the near-zero values
  published, and it costs more accuracy.
- NIFTY on German: **not reproduced**. The published German result is very close to what a model that approves *everyone*
  would score (see the report).

**Fair comparison** (same rules for every method):

- Using the graph raises accuracy on the real Pokec graphs (AUC about 70 → 76). It also makes the sensitive attribute
  more recoverable from the model's scores.
- **FairGNN** clearly reduces the parity gap where its estimator can actually guess the sensitive attribute (German,
  Pokec), at a cost of up to about 2–4 AUC points. It has no effect on Credit.
- **NIFTY** reliably achieves *its own* goals (counterfactual unfairness about 1% or less, instability cut by a third to three-quarters). But it does not
  significantly reduce the group parity gap compared with a plain GCN that never sees the sensitive attribute.
  Flipping a person's own attribute does not change the friends they are connected to, and on Pokec those friends are
  where the bias comes from.

**Robustness note:** we first ran the MLP vs. GCN comparison on a different random data split (`results/archive_seed19/`).
One finding that was statistically significant on that split was not significant on the official split. Conclusions drawn from
a single split, which is the standard practice in this literature, can be fragile.

---

## 9. Differences from the original code (for transparency)

- Both baselines were **re-written in plain PyTorch**, following the authors' released code line by line. The originals
  need old library versions (DGL 0.4.3, Python 3.7-era PyTorch Geometric) that no longer install easily on Colab.
  Intentional quirks of the original code are kept and marked with comments.
- All features are scaled to [−1, 1] on every dataset. The original code used raw features for Pokec and German.
- Every method is run with 5 random seeds (0–4). The authors' scripts use a single seed.
- Each method is evaluated twice: with **its own model-selection rule** (to reproduce the papers) and with **one common
  rule** for all methods (best validation AUC, plus a decision threshold chosen on validation data for balanced accuracy).

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `device: cpu` is printed in Cell 4 | You forgot to enable the GPU (Section 3, step 2). It still works, just much slower. |
| Colab disconnected halfway | Re-run Cells 0–5 and 9–11, then the cell that was running. Finished runs are skipped automatically. |
| `FileNotFoundError ... m3_results_v4.csv` in Cell 15 | Cell 5 has not finished yet. Cell 15 needs both Cell 5 and Cell 13 results. |
| "You have exhausted your GPU quota" | Colab's free GPU has daily limits. Wait a day or use Kaggle Notebooks (free GPU hours each week). |
| Numbers differ slightly from the tables here | Expected: GPU sparse operations are not bit-for-bit reproducible. Differences should be within the ± values shown. |

---

## 11. References

- E. Dai and S. Wang. *Say No to the Discrimination: Learning Fair Graph Neural Networks with Limited Sensitive Attribute Information.* WSDM 2021. [arXiv:2009.01454](https://arxiv.org/abs/2009.01454)
- C. Agarwal, H. Lakkaraju and M. Zitnik. *Towards a Unified Framework for Fair and Stable Graph Representation Learning.* UAI 2021. [arXiv:2102.13186](https://arxiv.org/abs/2102.13186)
- T. N. Kipf and M. Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.

Datasets and original code belong to their authors. The NIFTY code is MIT-licensed. The FairGNN repository does not state a
license, so its data is linked here, not copied.
