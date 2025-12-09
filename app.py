# app.py
# -------------------------------
# Import required libraries
# -------------------------------
import streamlit as st          # For web app UI
import pandas as pd            # For data handling
import numpy as np             # For numerical operations
from sklearn.preprocessing import MinMaxScaler  # For normalization
import plotly.graph_objects as go  # For radar and bar charts
import plotly.express as px        # For simpler charts like boxplots

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    layout="wide",               # Use wide layout for better visualization
    page_title="Squad Formation Assistant"  # Browser tab title
)

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data  # Cache to avoid reloading CSV on every interaction
def load_df(path="football_players_dataset.csv"):
    df = pd.read_csv(path)  # Read CSV into pandas DataFrame
    # Convert synergy list string into Python list
    df["synergy_list"] = df["synergy_list"].fillna("").apply(lambda x: x.split(",") if x else [])
    return df

df_all = load_df()  # Load the dataset once

# -------------------------------
# Identify numeric attributes
# -------------------------------
NUM_ATTRS = df_all.select_dtypes(include=np.number).columns.tolist()  # Select numeric columns
NUM_ATTRS = [col for col in NUM_ATTRS if col not in ["player_id","overall_rating"]]  # Exclude IDs & overall_rating

# -------------------------------
# Sidebar: Coach selects desired attributes
# -------------------------------
st.sidebar.title("Coach Attribute Selection")

# Toggle all attributes using session state
if "all_attrs_selected" not in st.session_state:
    st.session_state.all_attrs_selected = True

# Button to select/deselect all attributes at once
if st.sidebar.button("Toggle All Attributes"):
    st.session_state.all_attrs_selected = not st.session_state.all_attrs_selected

# Dictionary to store selected attributes and their desired values
selected_attrs = {}
for attr in NUM_ATTRS:
    use_attr = st.sidebar.checkbox(f"Use {attr}", value=st.session_state.all_attrs_selected)
    if use_attr:
        col_min, col_max = df_all[attr].min(), df_all[attr].max()
        # For normalized attributes (0-1), use slider
        if 0 <= col_min and col_max <= 1:
            val = st.sidebar.slider(f"Desired {attr}", float(col_min), float(col_max),
                                    float((col_min+col_max)/2), step=0.01)
        # For regular numeric attributes, use number input
        else:
            val = st.sidebar.number_input(f"Desired {attr}", int(col_min), int(col_max),
                                          int((col_min+col_max)/2))
        selected_attrs[attr] = val

# If no attribute is selected, pick first 5 numeric attributes as default
if not selected_attrs:
    for attr in NUM_ATTRS[:5]:
        selected_attrs[attr] = int(df_all[attr].mean())

SCORING_ATTRS = list(selected_attrs.keys())  # Attributes to use for scoring

# -------------------------------
# Normalize coach input
# -------------------------------
req_df = pd.DataFrame([selected_attrs])  # Convert selected attributes into DataFrame
scaler = MinMaxScaler()                  # Min-max scaling to 0-1
scaler.fit(df_all[SCORING_ATTRS])        # Fit on dataset
req_norm = scaler.transform(req_df[SCORING_ATTRS])[0]  # Transform coach input

# -------------------------------
# Select meaningful attributes for visualization
# -------------------------------
MEANINGFUL_ATTRS = [attr for attr in ["pace","shooting","passing","defending","stamina"] if attr in df_all.columns]
if len(MEANINGFUL_ATTRS) < 5:
    # Fill up to 5 with scoring attributes if some default ones are missing
    MEANINGFUL_ATTRS += [a for a in SCORING_ATTRS if a not in MEANINGFUL_ATTRS][:5-len(MEANINGFUL_ATTRS)]

# -------------------------------
# Session state: selected players
# -------------------------------
if "selected" not in st.session_state:
    st.session_state.selected = []

selected_ids = st.session_state.selected
# Candidate players are those not already selected
candidate_df = df_all[~df_all["player_id"].isin(selected_ids)].copy()

# -------------------------------
# Collaborative boost
# -------------------------------
use_collab = st.sidebar.checkbox("Use synergy", True)  # Include synergy effect
w_collab = st.sidebar.slider("Collaborative weight", 0.0, 1.0, 0.25)  # Weight of collaboration

# Button to reset squad
if st.sidebar.button("Reset Squad"):
    st.session_state.selected = []

# -------------------------------
# Vectorized scoring
# -------------------------------
# Normalize candidates
candidate_norm = pd.DataFrame(scaler.transform(candidate_df[SCORING_ATTRS]), columns=SCORING_ATTRS)
norms_candidates = np.linalg.norm(candidate_norm.values, axis=1)  # L2 norm of each candidate
norm_req = np.linalg.norm(req_norm)  # L2 norm of coach input
# Cosine similarity for content score
cos_sim = candidate_norm.values.dot(req_norm) / (norms_candidates * norm_req)
candidate_df["content_score"] = cos_sim

# Collaborative score based on synergy
if selected_ids and use_collab:
    candidate_df["collab_score"] = candidate_df["synergy_list"].apply(
        lambda x: len(set(x).intersection(selected_ids))/max(1,len(x))
    )
else:
    candidate_df["collab_score"] = 0

# Final weighted score
candidate_df["final_score"] = 0.75 * candidate_df["content_score"] + w_collab * candidate_df["collab_score"]

# Sort candidates by final score
candidate_df = candidate_df.sort_values("final_score", ascending=False).reset_index(drop=True)

# -------------------------------
# Multi-select filter by position
# -------------------------------
all_positions = df_all["position"].unique().tolist()
selected_positions = st.multiselect(
    "Select positions to display (leave empty for all)", 
    options=all_positions,
    default=all_positions
)

# Filter candidates and full dataframe by positions
filtered_candidates = candidate_df[candidate_df["position"].isin(selected_positions)] if selected_positions else candidate_df
filtered_df_all = df_all[df_all["position"].isin(selected_positions)] if selected_positions else df_all

# -------------------------------
# Filter by minimum final score
# -------------------------------
score_threshold = st.sidebar.slider("Minimum Final Score", 0.0, 1.0, 0.5, 0.01)
filtered_candidates = filtered_candidates[filtered_candidates["final_score"] >= score_threshold]

# -------------------------------
# Display Current Squad
# -------------------------------
st.subheader("Current Squad")
if selected_ids:
    st.table(filtered_df_all[filtered_df_all["player_id"].isin(selected_ids)][
        ["player_id","name","position","overall_rating"]
    ])
else:
    st.write("No players selected yet.")

# -------------------------------
# Display Top Recommendations
# -------------------------------
st.subheader("Top Recommendations")
top_k = 10
if filtered_candidates.empty:
    st.write("No candidates found for the selected positions/attributes/score threshold.")
else:
    st.table(filtered_candidates.head(top_k)[
        ["player_id","name","position","overall_rating","final_score"]
    ])

# -------------------------------
# Add player to squad (fixed double-tap)
# -------------------------------
st.subheader("Add Player to Squad")
pick = st.selectbox(
    "Pick player to add",
    options=[""] + filtered_candidates.head(50)["player_id"].tolist()
)
# Prevent adding the same player twice
if pick and pick not in st.session_state.selected:
    if st.button("Add Player"):
        st.session_state.selected.append(pick)
        st.success(f"Player {pick} added to squad!")

# -------------------------------
# 1. Radar chart: Top candidate vs squad average
# -------------------------------
def plot_radar(player_row, squad_df):
    attrs = MEANINGFUL_ATTRS
    player_vals = player_row[attrs].values
    squad_avg = squad_df[attrs].mean().tolist() if not squad_df.empty else [0]*len(attrs)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=player_vals, theta=attrs, fill='toself', name=player_row["name"], line=dict(color='red')
    ))
    fig.add_trace(go.Scatterpolar(
        r=squad_avg, theta=attrs, fill='toself', name="Squad Avg", line=dict(color='blue')
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title="Radar: Top Candidate vs Squad Avg")
    return fig

if not filtered_candidates.empty:
    st.plotly_chart(plot_radar(filtered_candidates.iloc[0],
                               filtered_df_all[filtered_df_all["player_id"].isin(selected_ids)]))

# -------------------------------
# 2. Stacked Bar: Content vs Collaboration
# -------------------------------
def plot_stacked_scores(ranked_df):
    df = ranked_df.copy().head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Content', x=df["name"], y=df["content_score"]*0.75, marker_color='green'))
    fig.add_trace(go.Bar(name='Collaborative', x=df["name"], y=df["collab_score"]*w_collab, marker_color='orange'))
    fig.update_layout(barmode='stack', xaxis_tickangle=-45, yaxis_title="Score",
                      title="Top 10 Player Scores Breakdown", legend_title="Component")
    return fig

if not filtered_candidates.empty:
    st.plotly_chart(plot_stacked_scores(filtered_candidates))

# -------------------------------
# 3. Attribute Comparison (Top 5)
# -------------------------------
def plot_attribute_comparison(ranked_df):
    df = ranked_df.copy().head(5)
    attrs = MEANINGFUL_ATTRS
    fig = go.Figure()
    colors = ['red','blue','green','orange','purple']
    for i, attr in enumerate(attrs):
        fig.add_trace(go.Bar(name=attr, x=df["name"], y=df[attr], marker_color=colors[i]))
    fig.update_layout(barmode='group', xaxis_tickangle=-45, yaxis_title="Attribute Value",
                      title="Top 5 Players Attribute Comparison")
    return fig

if not filtered_candidates.empty:
    st.plotly_chart(plot_attribute_comparison(filtered_candidates))

# -------------------------------
# 4. Attribute Distribution by Position
# -------------------------------
st.subheader("Attribute Distribution by Position")
for attr in MEANINGFUL_ATTRS:
    fig = px.box(filtered_df_all, x="position", y=attr, color="position",
                 title=f"{attr.capitalize()} Distribution by Position")
    st.plotly_chart(fig)

# -------------------------------
# 5. Top Candidates by Position
# -------------------------------
st.subheader("Top Candidates by Position")
if not filtered_candidates.empty:
    top_positions = filtered_candidates.groupby("position").head(5)
    fig = px.bar(top_positions, x="name", y="final_score", color="position",
                 hover_data=["overall_rating", "content_score", "collab_score"],
                 title="Top Candidates by Position")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
else:
    st.write("No candidates to display by position/score threshold.")
