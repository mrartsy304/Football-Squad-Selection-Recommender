# -------------------------------
# Import required libraries
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    layout="wide",
    page_title="Squad Formation Assistant",
)

# -------------------------------
# Load dataset (synergy_list as tuple to remove Streamlit hash warning)
# -------------------------------
@st.cache_data
def load_df(path="football_players_dataset.csv"):
    df = pd.read_csv(path)
    df["synergy_list"] = df["synergy_list"].fillna("").apply(lambda x: tuple(x.split(",")) if x else ())
    return df

df_all = load_df()

# -------------------------------
# Safe Cosine Similarity (vectorized)
# -------------------------------
def safe_cosine_vectorized(candidate_matrix, req_vector):
    norm_candidates = np.linalg.norm(candidate_matrix, axis=1)
    norm_req = np.linalg.norm(req_vector)
    dot_products = candidate_matrix @ req_vector
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_sim = np.divide(dot_products, norm_candidates * norm_req)
        cos_sim[np.isnan(cos_sim)] = 0.0
    return cos_sim

# -------------------------------
# Identify numeric attributes
# -------------------------------
NUM_ATTRS = df_all.select_dtypes(include=np.number).columns.tolist()
NUM_ATTRS = [col for col in NUM_ATTRS if col not in ["player_id","overall_rating"]]

# -------------------------------
# Sidebar: Coach Attribute Selection
# -------------------------------
st.sidebar.title("🏋️ Coach Attribute Selection")
if "all_attrs_selected" not in st.session_state:
    st.session_state.all_attrs_selected = True

with st.sidebar.expander("Attribute Selection", expanded=True):
    if st.button("Toggle All Attributes"):
        st.session_state.all_attrs_selected = not st.session_state.all_attrs_selected

    selected_attrs = {}
    for attr in NUM_ATTRS:
        use_attr = st.checkbox(f"Use {attr}", value=st.session_state.all_attrs_selected)
        if use_attr:
            col_min, col_max = df_all[attr].min(), df_all[attr].max()
            if 0 <= col_min and col_max <= 1:
                val = st.slider(f"Desired {attr}", float(col_min), float(col_max),
                                float((col_min+col_max)/2), step=0.01)
            else:
                val = st.number_input(f"Desired {attr}", int(col_min), int(col_max),
                                      int((col_min+col_max)/2))
            selected_attrs[attr] = val

if not selected_attrs:
    for attr in NUM_ATTRS[:5]:
        selected_attrs[attr] = int(df_all[attr].mean())

SCORING_ATTRS = list(selected_attrs.keys())

# -------------------------------
# Normalize coach input
# -------------------------------
@st.cache_data
def get_scaler(df, attrs):
    scaler = MinMaxScaler()
    scaler.fit(df[attrs])
    return scaler

scaler = get_scaler(df_all, SCORING_ATTRS)
req_norm = scaler.transform(pd.DataFrame([selected_attrs]))[0]

# -------------------------------
# Meaningful attributes for visualization
# -------------------------------
MEANINGFUL_ATTRS = [attr for attr in ["pace","shooting","passing","defending","stamina"] if attr in df_all.columns]
if len(MEANINGFUL_ATTRS) < 5:
    MEANINGFUL_ATTRS += [a for a in SCORING_ATTRS if a not in MEANINGFUL_ATTRS][:5-len(MEANINGFUL_ATTRS)]

# -------------------------------
# Session state: selected players
# -------------------------------
if "selected" not in st.session_state:
    st.session_state.selected = []

selected_ids = st.session_state.selected
candidate_df = df_all[~df_all["player_id"].isin(selected_ids)].copy()

# -------------------------------
# Collaborative & Team Boost
# -------------------------------
with st.sidebar.expander("🤝 Collaboration & Team Boost", expanded=True):
    use_collab = st.checkbox("Use Synergy", True)
    w_collab = st.slider("Synergy Weight", 0.0, 1.0, 0.25)
    use_team = st.checkbox("Use Team Boost", True)
    team_boost_weight = st.slider("Team Boost Weight", 0.0, 1.0, 0.25)
    if st.button("🔄 Reset Squad"):
        st.session_state.selected = []

# -------------------------------
# Vectorized scoring
# -------------------------------
candidate_norm = pd.DataFrame(scaler.transform(candidate_df[SCORING_ATTRS]), columns=SCORING_ATTRS)
candidate_df["content_score"] = safe_cosine_vectorized(candidate_norm.values, req_norm)

# -------------------------------
# Synergy score
# -------------------------------
if selected_ids and use_collab:
    candidate_df["collab_score"] = candidate_df["synergy_list"].apply(
        lambda x: len(set(x).intersection(selected_ids))/max(1,len(x))
    )
else:
    candidate_df["collab_score"] = 0

# -------------------------------
# Exponential Team Boost
# -------------------------------
if selected_ids and use_team:
    squad_team_counts = df_all[df_all["player_id"].isin(selected_ids)]["team_name"].value_counts().to_dict()
    def exp_team_score(team_name):
        count = squad_team_counts.get(team_name, 0)
        return 0.1 * (2**count - 1)
    candidate_df["team_score"] = candidate_df["team_name"].apply(exp_team_score)
else:
    candidate_df["team_score"] = 0

# -------------------------------
# Final score (rounded to 2 decimals)
# -------------------------------
candidate_df["final_score"] = (
    0.75 * candidate_df["content_score"] +
    w_collab * candidate_df["collab_score"] +
    team_boost_weight * candidate_df["team_score"]
).round(2)

# -------------------------------
# Count requirements matched (tolerance 5%)
# -------------------------------
attr_ranges = df_all[SCORING_ATTRS].max() - df_all[SCORING_ATTRS].min()
tolerance_matrix = np.abs(candidate_df[SCORING_ATTRS] - pd.Series(selected_attrs)) <= 0.05 * attr_ranges
candidate_df["num_req_matched"] = tolerance_matrix.sum(axis=1)

# -------------------------------
# Option B Sorting: final_score primary, num_req_matched secondary
# -------------------------------
candidate_df = candidate_df.sort_values(
    ["final_score", "num_req_matched", "overall_rating"],
    ascending=[False, False, False]
).reset_index(drop=True)

# -------------------------------
# Filter by positions & score
# -------------------------------
all_positions = df_all["position"].unique().tolist()
selected_positions = st.multiselect("Select positions (empty = all)", options=all_positions, default=all_positions)
filtered_candidates = candidate_df[candidate_df["position"].isin(selected_positions)] if selected_positions else candidate_df
filtered_df_all = df_all[df_all["position"].isin(selected_positions)] if selected_positions else df_all

score_threshold = st.sidebar.slider("Minimum Final Score", 0.0, 1.0, 0.5, 0.01)
filtered_candidates = filtered_candidates[filtered_candidates["final_score"] >= score_threshold]

# -------------------------------
# Page Layout: Squad & Recommendations (Top 10)
# -------------------------------
st.title("⚽ Squad Formation Assistant")
st.markdown("Build your dream football squad with synergy, team compatibility & stats!")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("🛡️ Current Squad")
    if selected_ids:
        squad_df = filtered_df_all[filtered_df_all["player_id"].isin(selected_ids)]
        for _, row in squad_df.iterrows():
            st.markdown(f"**{row['name']}** ({row['position']}) - ⭐ {row['overall_rating']}")
            st.progress(row['overall_rating']/100)
    else:
        st.info("No players selected yet.")

with col2:
    st.subheader("🏆 Top Recommendations")
    top_k = 10
    if filtered_candidates.empty:
        st.warning("No candidates match your filters!")
    else:
        for _, row in filtered_candidates.head(top_k).iterrows():
            card_col1, card_col2 = st.columns([3,1])
            with card_col1:
                st.markdown(
                    f"**{row['name']}** ({row['position']}) - ⭐ {row['overall_rating']} | "
                    f"Score: {row['final_score']:.2f} | "
                    f"Content: {row['content_score']:.2f} | "
                    f"Collab: {row['collab_score']:.2f} | "
                    f"Team: {row['team_score']:.2f} | "
                    f"Req Matched: {row['num_req_matched']}/{len(selected_attrs)}"
                )
            with card_col2:
                st.button("➕ Add", key=row['player_id'],
                          on_click=lambda x=row['player_id']: st.session_state.selected.append(x))

# -------------------------------
# Squad Metrics
# -------------------------------
if selected_ids:
    st.subheader("📊 Squad Average Metrics")
    squad_stats = filtered_df_all[filtered_df_all["player_id"].isin(selected_ids)][MEANINGFUL_ATTRS].mean()
    metrics_cols = st.columns(len(MEANINGFUL_ATTRS))
    for i, attr in enumerate(MEANINGFUL_ATTRS):
        metrics_cols[i].metric(attr.capitalize(), f"{squad_stats[attr]:.1f}")

# -------------------------------
# Visualizations
# -------------------------------
if not filtered_candidates.empty:
    st.subheader("📈 Visualizations")

    # Radar chart
    def plot_radar(player_row, squad_df):
        attrs = MEANINGFUL_ATTRS
        player_vals = player_row[attrs].values
        squad_avg = squad_df[attrs].mean().tolist() if not squad_df.empty else [0]*len(attrs)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=player_vals, theta=attrs, fill='toself',
                                      name=player_row["name"], line=dict(color='red')))
        fig.add_trace(go.Scatterpolar(r=squad_avg, theta=attrs, fill='toself',
                                      name="Squad Avg", line=dict(color='blue')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          showlegend=True, title="Radar: Top Candidate vs Squad Avg")
        return fig

    st.plotly_chart(
        plot_radar(
            filtered_candidates.iloc[0],
            filtered_df_all[filtered_df_all["player_id"].isin(selected_ids)]
        ),
        use_container_width=True
    )

    # Stacked score bar
    def plot_stacked_scores(ranked_df):
        df = ranked_df.copy().head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Content', x=df["name"], y=df["content_score"]*0.75))
        fig.add_trace(go.Bar(name='Collaboration', x=df["name"], y=df["collab_score"]*w_collab))
        fig.add_trace(go.Bar(name='Team Boost', x=df["name"], y=df["team_score"]*team_boost_weight))
        fig.update_layout(barmode='stack', xaxis_tickangle=-45,
                          yaxis_title="Score", title="Top 10 Player Scores Breakdown")
        return fig
    st.plotly_chart(plot_stacked_scores(filtered_candidates), use_container_width=True)

    # Attribute Comparison
    def plot_attribute_comparison(ranked_df):
        df = ranked_df.copy().head(5)
        attrs = MEANINGFUL_ATTRS
        fig = go.Figure()
        for attr in attrs:
            fig.add_trace(go.Bar(name=attr, x=df["name"], y=df[attr]))
        fig.update_layout(barmode='group', xaxis_tickangle=-45,
                          yaxis_title="Attribute Value", title="Top 5 Players Attribute Comparison")
        return fig
    st.plotly_chart(plot_attribute_comparison(filtered_candidates), use_container_width=True)

# -------------------------------
# Attribute Distribution by Position
# -------------------------------
with st.expander("📦 Attribute Distribution by Position"):
    for attr in MEANINGFUL_ATTRS:
        fig = px.box(filtered_df_all, x="position", y=attr, color="position",
                     title=f"{attr.capitalize()} Distribution by Position")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Top Candidates by Position
# -------------------------------
with st.expander("🥇 Top Candidates by Position"):
    if not filtered_candidates.empty:
        top_positions = filtered_candidates.groupby("position").head(5)
        fig = px.bar(top_positions, x="name", y="final_score", color="position",
                     hover_data=["overall_rating","content_score","collab_score"],
                     title="Top Candidates by Position")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
