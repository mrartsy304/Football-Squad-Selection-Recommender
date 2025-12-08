# 🏟️ Squad Formation Assistant

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24-orange)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Project Overview

**Squad Formation Assistant** is an interactive **Streamlit web app** designed to help football coaches select and form the best squad using **data-driven analytics**.  

It recommends players based on:  
- Desired player attributes  
- Position filtering  
- Collaborative synergy with already selected players  

It also provides **visual insights** via radar charts, stacked bars, attribute comparisons, and position-based distributions.

---

## 🎯 Project Objectives

1. **Attribute Matching:** Identify players whose stats align with coach preferences.  
2. **Collaborative Synergy:** Evaluate player compatibility with already selected squad members.  
3. **Interactive Selection:** Dynamically add/remove players to the squad.  
4. **Visual Analytics:** Compare player and squad metrics visually.

---

## 📂 Dataset

**File:** `football_players_level3_10000.csv`  

| Column | Description |
|--------|-------------|
| `player_id` | Unique player identifier |
| `name` | Player name |
| `position` | Player position (GK, CB, ST, etc.) |
| `overall_rating` | Overall rating of the player |
| `synergy_list` | List of player IDs that synergize well |
| `pace, shooting, passing, defending, stamina, ...` | Numeric player attributes used for scoring |

**Notes:**  
- `synergy_list` is used for collaborative scoring.  
- Numeric attributes are normalized before calculating similarity.

---

## ⚙️ Project Logic & Working

### 1️⃣ User Input
- Coach selects desired player attributes via sidebar.  
- "Toggle All Attributes" button selects/deselects all attributes.  
- Minimum final score can be set to filter recommendations.

### 2️⃣ Candidate Filtering
- Removes already selected players from candidates.  
- Filter candidates by position using multi-select.  
- Optional minimum final score filter applied.

### 3️⃣ Scoring Logic

**a) Content-based Score (Cosine Similarity):**

```text
content_score = (candidate_attributes ⋅ desired_attributes) / (||candidate|| * ||desired||)
