# 🏟️ Squad Formation Assistant
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
```

**b) Collaborative Synergy Score:**
```text
collab_score = len(synergy_list ∩ selected_ids) / max(1, len(synergy_list))
```

**c)Final Score:**
```text
final_score = 0.75 * content_score + w_collab * collab_score
```

### 4️⃣ Player Selection
- Select a player from top recommendations.
- Added players are saved in session state to prevent duplicates.
- "Reset Squad" button clears all selected players.

### 5️⃣ Visualizations
- Radar Chart: Candidate vs Squad Average
- Stacked Bar: Content vs Collaboration Score
- Top 5 Attribute Comparison: Grouped bar chart
- Boxplots: Attribute distribution by position
- Top Candidates by Position: Top 5 candidates per position

### Code Structure
- app.py — Main Streamlit app
- football_players_level3_10000.csv — Dataset
- requirements.txt — Dependencies
# Key Sections in app.py:
- Load dataset & cache with st.cache_data
- Sidebar for attribute selection & toggle
- Normalize attributes with MinMaxScaler
- Session state for selected players
- Compute content, collaboration, final scores
- Filter candidates by position & score
- Display current squad & top recommendations
- Add player (prevents double-tap)
- Visualizations: Radar, stacked bar, attribute comparison, distributions, top candidates

  <img width="1010" height="623" alt="image" src="https://github.com/user-attachments/assets/bbd57879-bb2a-4dec-925c-e1aec0e588b1" />

  <img width="998" height="368" alt="image" src="https://github.com/user-attachments/assets/5f08e733-b7c2-4bc6-905d-f753890347a4" />

