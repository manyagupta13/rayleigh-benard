# Queue Waiting Time Prediction and Distribution Modeling

## 📘 Overview
This project explores **queue waiting time behavior** using real-world queue data. It performs data preprocessing, exploratory analysis, and probabilistic modeling to determine which distribution — **Exponential** or **Gamma** — best describes the waiting time characteristics.

The notebook integrates **Maximum Likelihood Estimation (MLE)**, **Bayesian inference**, and **hypothesis testing** to statistically validate the model selection.

---

## 📊 Dataset
- **Source**: Imported from multiple CSV files under `/kaggle/input/`.
- **Records**: 560 entries after cleaning.
- **Columns**:
  | Column | Description |
  |---------|-------------|
  | `arrival_time` | Time when customer/task arrives |
  | `start_time` | Time when service starts |
  | `finish_time` | Time when service finishes |
  | `wait_time` | Waiting duration in seconds |
  | `queue_length` | Queue size at the time of arrival |

---

## 🔍 Analysis Workflow

### 1. **Data Preprocessing**
- Merged multiple CSV files into a single DataFrame.  
- Removed invalid or negative wait times.  
- Converted time columns (`arrival_time`, `start_time`, `finish_time`) to datetime objects.  
- Generated new time-based features such as `hour`.

### 2. **Exploratory Data Analysis (EDA)**
- Visualized **queue length over time** and **waiting time variation by hour**.  
- Created **boxplots** showing how waiting time changes with queue length.  
- Computed descriptive statistics for all numerical features.  
- Generated **correlation heatmaps** to identify relationships among variables.

### 3. **Statistical Modeling**
#### a. **Exponential Distribution (MLE)**
- Estimated λ (rate parameter) using MLE:  
  \[
  \hat{\lambda} = \frac{1}{\bar{x}} = 0.0964
  \]
- Fitted PDF visualized alongside the empirical distribution.

#### b. **Gamma Distribution (MLE & Refinement)**
- Estimated parameters using moment-based and iterative refinement methods:  
  \[
  k = 3.5564, \quad \theta = 2.9121
  \]
- Validated with `scipy.stats.gamma.fit()` for consistency.

#### c. **Bayesian Inference**
- Applied Gamma prior over λ for the Exponential model.  
- Derived posterior distribution:
  \[
  \text{Posterior Mean of } \lambda = 0.0967
  \]
- Visualized **prior vs. posterior** for interpretability.

#### d. **Model Comparison**
- Conducted **Likelihood Ratio Test (LRT)**:
  ```
  LRT Statistic: 309.50, p-value < 0.0001
  → Reject Null Hypothesis (Gamma fits better)
  ```
- Calculated **Bayesian Information Criterion (BIC)**:
  ```
  BIC (Exponential): 3659.91  
  BIC (Gamma): 3356.71  
  → Gamma model preferred
  ```

---

## 🧠 Key Insights
- Waiting times follow a **Gamma-like distribution**, indicating variability in queue dynamics beyond the simple exponential assumption.  
- Queue length shows **moderate correlation** with waiting time, suggesting both time-dependent and stochastic effects.  
- Bayesian estimation provides interpretable uncertainty over rate parameters.

---

## ⚙️ Tools & Libraries
- **Python 3.x**
- **NumPy**, **Pandas** – data handling  
- **Matplotlib**, **Seaborn** – visualization  
- **SciPy** – statistical inference  
- **scikit-learn** – preprocessing  

---

## 🚀 How to Run
1. Clone the repository or open the notebook in Kaggle.  
2. Place all CSV data files inside the `/input` directory.  
3. Run all cells in sequence — outputs include visualizations, fitted parameters, and statistical summaries.

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

---

## 📈 Results Summary
| Model | Parameters | Log-Likelihood | BIC | Verdict |
|--------|-------------|----------------|------|----------|
| Exponential | λ = 0.0966 | -1829.95 | 3659.91 | ❌ Rejected |
| Gamma | k = 4.84, θ = 2.14 | -1676.36 | 3356.71 | ✅ Preferred |

---

## 🧩 Future Work
- Extend analysis to **predict future waiting times** using regression or time-series models.  
- Integrate **queue simulation models** for real-time service optimization.  
- Explore **hierarchical Bayesian models** for multi-queue systems.

---

## ✍️ Author
**Manya Gupta**  
Data Science & AI Research Enthusiast  
📫 [LinkedIn or GitHub link, if you’d like to include]
