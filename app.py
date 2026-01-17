import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as path_effects
import os
import random

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Wellbeing Dashboard")

# --- Constants & Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "wellbeing_surveys.csv")
ICON_DIR = os.path.join(BASE_DIR, "icons")

COLORS = {
    'green': '#2ecc71',
    'orange': '#f39c12',
    'red': '#e74c3c',
    'grey': '#bdc3c7'
}

GROUPED_DIMENSIONS = {
    'Body': ['FRUITS_VEGGIES', 'BMI_RANGE_neg', 'DAILY_STEPS', 'SLEEP_HOURS'],
    'Mind': ['DAILY_STRESS_neg', 'FLOW', 'WEEKLY_MEDITATION', 'DAILY_SHOUTING_neg'],
    'Expertise': ['ACHIEVEMENT', 'PERSONAL_AWARDS', 'TIME_FOR_PASSION', 'TODO_COMPLETED'],
    'Connection': ['PLACES_VISITED', 'CORE_CIRCLE', 'SUPPORTING_OTHERS', 'SOCIAL_NETWORK'],
    'Purpose': ['DONATION', 'LOST_VACATION_neg', 'SUFFICIENT_INCOME', 'LIVE_VISION'],
}

DIMENSIONS = list(GROUPED_DIMENSIONS.keys())

QUESTIONS_DICT = {
    'Body_FRUITS_VEGGIES': "Quantes racions de fruita o verdura menges cada dia?",
    'Body_BMI_RANGE_neg': "Quin és el teu rang d'Índex de Massa Corporal (IMC)?",
    'Body_DAILY_STEPS': "Quants milers de passos acostumes a caminar cada dia?",
    'Body_SLEEP_HOURS': "Aproximadament, quantes hores acostumes a dormir cada dia?",
    'Mind_DAILY_STRESS_neg': "Quin nivell d'estrès experimentes habitualment cada dia?",
    'Mind_FLOW': "En un dia normal, quantes hores experimentes l'estat de 'flow' (concentració absoluta i gaudi)?",
    'Mind_WEEKLY_MEDITATION': "En una setmana normal, quantes vegades tens l'oportunitat de reflexionar sobre tu mateix/a o meditar?",
    'Mind_DAILY_SHOUTING_neg': "Amb quina freqüència crides o critiques a algú en una setmana normal?",
    'Expertise_ACHIEVEMENT': "De quants assoliments destacables et sents orgullós/osa?",
    'Expertise_PERSONAL_AWARDS': "Quants reconeixements o premis has rebut al llarg de la teva vida?",
    'Expertise_TIME_FOR_PASSION': "Quantes hores dediques cada dia a fer allò que t'apassiona?",
    'Expertise_TODO_COMPLETED': "Fins a quin punt completes les teves llistes de tasques setmanals?",
    'Connection_PLACES_VISITED': "Quants llocs nous visites cada any?",
    'Connection_CORE_CIRCLE': "Quantes persones formen part del teu cercle més proper (família i amics íntims)?",
    'Connection_SUPPORTING_OTHERS': "A quantes persones ajudes actualment a tenir una vida millor? (comptant l'últim any)",
    'Connection_SOCIAL_NETWORK': "Amb quantes persones interactues durant un dia normal?",
    'Purpose_DONATION': "Quantes vegades dones el teu temps o diners a bones causes? (en un any normal)",
    'Purpose_LOST_VACATION_neg': "Quants dies de vacances acostumes a 'perdre' (no gaudir) cada any?",
    'Purpose_SUFFICIENT_INCOME': "Els teus ingressos són suficients per cobrir les despeses bàsiques de la vida?",
    'Purpose_LIVE_VISION': "Amb quants anys d'antelació tens clara la teva visió de vida o els teus objectius personals?"
}

DISCRETE_VALUES_MAP = {
    'Body_FRUITS_VEGGIES': [0, 1, 2, 3, 4, 5],
    'Body_BMI_RANGE_neg': [0, 1],
    'Body_DAILY_STEPS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Body_SLEEP_HOURS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mind_DAILY_STRESS_neg': [0, 1, 2, 3, 4, 5],
    'Mind_FLOW': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mind_WEEKLY_MEDITATION': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mind_DAILY_SHOUTING_neg': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Expertise_ACHIEVEMENT': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Expertise_PERSONAL_AWARDS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Expertise_TIME_FOR_PASSION': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Expertise_TODO_COMPLETED': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Connection_PLACES_VISITED': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Connection_CORE_CIRCLE': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Connection_SUPPORTING_OTHERS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Connection_SOCIAL_NETWORK': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Purpose_DONATION': [0, 1, 2, 3, 4, 5],
    'Purpose_LOST_VACATION_neg': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Purpose_SUFFICIENT_INCOME': [0, 1],
    'Purpose_LIVE_VISION': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

EMOJI_MAP = {
    'Body_FRUITS_VEGGIES': '🍎',
    'Body_BMI_RANGE_neg': '🍎',
    'Body_DAILY_STEPS': '🍎',
    'Body_SLEEP_HOURS': '🍎',
    'Mind_DAILY_STRESS_neg': '⏱️',
    'Mind_FLOW': '⏱️',
    'Mind_WEEKLY_MEDITATION': '⏱️',
    'Mind_DAILY_SHOUTING_neg': '⏱️',
    'Expertise_ACHIEVEMENT': '🏆',
    'Expertise_PERSONAL_AWARDS': '🏆',
    'Expertise_TIME_FOR_PASSION': '🏆',
    'Expertise_TODO_COMPLETED': '🏆',
    'Connection_PLACES_VISITED': '🫂',
    'Connection_CORE_CIRCLE': '🫂',
    'Connection_SUPPORTING_OTHERS': '🫂',
    'Connection_SOCIAL_NETWORK': '🫂',
    'Purpose_DONATION': '💰',
    'Purpose_LOST_VACATION_neg': '💰',
    'Purpose_SUFFICIENT_INCOME': '💰',
    'Purpose_LIVE_VISION': '💰'
}

PLACEHOLDER_MAP = {
    'Body_FRUITS_VEGGIES': 'racions al dia',
    'Body_BMI_RANGE_neg': '0 per <25, 1 per >25',
    'Body_DAILY_STEPS': 'milers de passos al dia',
    'Body_SLEEP_HOURS': 'hores al dia',
    'Mind_DAILY_STRESS_neg': '0 (gens) a 5 (molt)',
    'Mind_FLOW': 'hores al dia',
    'Mind_WEEKLY_MEDITATION': 'vegades a la setmana',
    'Mind_DAILY_SHOUTING_neg': '0 (mai) a 10 (sovint)',
    'Expertise_ACHIEVEMENT': 'de 0 a 10',
    'Expertise_PERSONAL_AWARDS': 'de 0 a 10',
    'Expertise_TIME_FOR_PASSION': 'hores al dia',
    'Expertise_TODO_COMPLETED': '0 (cap) a 10 (totes)',
    'Connection_PLACES_VISITED': 'de 0 a 10',
    'Connection_CORE_CIRCLE': 'de 0 a 10',
    'Connection_SUPPORTING_OTHERS': 'de 0 a 10',
    'Connection_SOCIAL_NETWORK': 'de 0 a 10',
    'Purpose_DONATION': 'de 0 a 5',
    'Purpose_LOST_VACATION_neg': 'de 0 a 10',
    'Purpose_SUFFICIENT_INCOME': '0 (No), 1 (Sí)',
    'Purpose_LIVE_VISION': 'anys en el futur'
}

ANSWERS_DISPLAY_MAP = {
    'Body_FRUITS_VEGGIES': "{} racions de fruita o verdura",
    'Body_BMI_RANGE_neg': "IMC: {}",
    'Body_DAILY_STEPS': "{} milers de passos",
    'Body_SLEEP_HOURS': "{} hores de son",
    'Mind_DAILY_STRESS_neg': "Nivell {} d'estrès",
    'Mind_FLOW': "{} hores de 'flow'",
    'Mind_WEEKLY_MEDITATION': "{} meditacions setmanals",
    'Mind_DAILY_SHOUTING_neg': "Freqüència de crits: {}/10",
    'Expertise_ACHIEVEMENT': "{} assoliments destacables",
    'Expertise_PERSONAL_AWARDS': "{} reconeixements",
    'Expertise_TIME_FOR_PASSION': "{} hores d'allò que t'apassiona",
    'Expertise_TODO_COMPLETED': "{}/10 tasques completades",
    'Connection_PLACES_VISITED': "{} llocs nous visitats",
    'Connection_CORE_CIRCLE': "{} persones al cercle íntim",
    'Connection_SUPPORTING_OTHERS': "Ajudes a {} persones",
    'Connection_SOCIAL_NETWORK': "Interactues amb {} persones",
    'Purpose_DONATION': "Dones {} vegades a l'any",
    'Purpose_LOST_VACATION_neg': "{} dies de vacances perduts",
    'Purpose_SUFFICIENT_INCOME': "Ingressos suficients: {}", 
    'Purpose_LIVE_VISION': "Visió a {} anys vista"
}

DIMENSION_TRANSLATIONS = {
    'Body':       'Cos',
    'Mind':       'Ment',
    'Expertise':  'Expertesa',
    'Connection': 'Connexió',
    'Purpose':    'Propòsit'
}

# --- 1. Data Preparation (Cached) ---
@st.cache_data
def load_and_process_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"Data file not found at: {CSV_PATH}")
        return None, None, None, None, None

    df_inicial = pd.read_csv(CSV_PATH)

    column_mapping = {
        col.replace('_neg', ''): f"{dim}_{col}"
        for dim, cols in GROUPED_DIMENSIONS.items()
        for col in cols
    }
    
    values_list = list(column_mapping.values())

    # Cleaning and Renaming
    df_int = (
        df_inicial
        .drop(columns=['Timestamp', 'WORK_LIFE_BALANCE_SCORE'], errors='ignore')
        .assign(DAILY_STRESS=lambda x: pd.to_numeric(x['DAILY_STRESS'], errors='coerce'))
        .dropna()
        .rename(columns=column_mapping)
    )

    # Adjustments
    if 'AGE' in df_int.columns:
        df_int['AGE'] = df_int['AGE'].replace('Less than 20', '20 or less')
    
    binary_cols = ['Body_BMI_RANGE_neg', 'Purpose_SUFFICIENT_INCOME']
    for col in binary_cols:
        if col in df_int.columns:
            df_int[col] = df_int[col] - 1

    score_cols = [col for col in df_int.columns if col not in ['AGE', 'GENDER']]
    df_int[score_cols] = df_int[score_cols].astype(int)

    # --- Calculations (NPS, Norm, Freq) ---
    nps_lookup = {}
    value_norm = {}
    value_freq = {}
    
    stats = df_int[values_list].agg(['mean', 'std', 'max'])
    n = len(df_int)

    for col in values_list:
        # NPS Lookup Logic
        mu, sigma, mx = stats.loc[['mean', 'std', 'max'], col]
        possible_vals = np.arange(int(mx) + 1)
        z = (possible_vals - mu) / sigma if sigma > 0 else np.zeros_like(possible_vals)
        raw_scores = (z > 0.5).astype(int) - (z < -0.5).astype(int)
        multiplier = -1 if '_neg' in col else 1
        nps_lookup[col] = (raw_scores * multiplier).tolist()

        # Norm Logic
        cdf = df_int[col].value_counts().sort_index().cumsum()
        norm_scores = cdf / n
        max_val = int(df_int[col].max())
        norm_scores = norm_scores.reindex(range(max_val + 1), method='ffill').fillna(0)
        scores_list = norm_scores.tolist()
        if '_neg' in col:
            scores_list = scores_list[::-1]
        value_norm[col] = [round(x, 4) for x in scores_list]

        # Freq Logic
        counts = df_int[col].value_counts().sort_index()
        counts = counts.reindex(range(max_val + 1), fill_value=0)
        value_freq[col] = counts.tolist()

    return df_int, nps_lookup, value_norm, value_freq, values_list

# --- 2. Icon Loading (Cached) ---
@st.cache_resource
def load_icons():
    icon_files = {
        'Body':       'icon_body.png',
        'Mind':       'icon_mind.png',
        'Expertise':  'icon_expertise.png',
        'Connection': 'icon_connection.png',
        'Purpose':    'icon_purpose.png'
    }
    loaded_icons = {}
    for dim, filename in icon_files.items():
        full_path = os.path.join(ICON_DIR, filename)
        try:
            img = mpimg.imread(full_path)
            loaded_icons[dim] = img
        except FileNotFoundError:
            loaded_icons[dim] = np.ones((60, 60, 3))
    return loaded_icons

# --- 3. Image Processing Logic ---
def get_partial_grayscale_icon(icon_array, ratio):
    processed = icon_array.copy()
    rows, cols, channels = processed.shape
    
    split_row = int(rows * (1 - ratio))
    
    top_section = processed[:split_row, :, :]
    
    if channels == 4:
        rgb = top_section[..., :3]
        gray_vals = np.dot(rgb, [0.2989, 0.5870, 0.1140])
        top_section[..., 0] = gray_vals
        top_section[..., 1] = gray_vals
        top_section[..., 2] = gray_vals
    else:
        gray_vals = np.dot(top_section[..., :3], [0.2989, 0.5870, 0.1140])
        top_section[..., 0] = gray_vals
        top_section[..., 1] = gray_vals
        top_section[..., 2] = gray_vals
        
    return processed

# --- 4. Plotting Logic ---

# --- 4a. Radar Chart Helper & Function ---
def draw_gradient_segment(ax, theta_start, r_start, theta_end, r_end, c_hex, alpha_start, alpha_end):
    n_points = 50
    t_vals = np.linspace(0, 1, n_points)

    x1, y1 = r_start * np.cos(theta_start), r_start * np.sin(theta_start)
    x2, y2 = r_end * np.cos(theta_end), r_end * np.sin(theta_end)

    xs = x1 * (1 - t_vals) + x2 * t_vals
    ys = y1 * (1 - t_vals) + y2 * t_vals
    # Convert back to polar for plotting
    rs = np.sqrt(xs**2 + ys**2)
    ts = np.arctan2(ys, xs)

    ts = np.unwrap(ts)

    points = np.array([ts, rs]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    rgb = mcolors.to_rgb(c_hex)
    alphas = np.linspace(alpha_start, alpha_end, n_points)
    rgba = np.zeros((n_points - 1, 4))
    rgba[:, :3] = rgb
    rgba[:, 3] = alphas[:-1]

    lc = LineCollection(segments, colors=rgba, linewidth=3)
    ax.add_collection(lc)

def create_radar_figure(user_row, value_norm, nps_lookup, loaded_icons):
    """
    Creates a Radar (Polar) chart summarizing the 5 dimensions.
    accepts user_row as either a Series or a dict.
    """
    dim_ranks = []
    dim_nps = []

    # Calculate aggregate scores for the 5 dimensions
    for dim in DIMENSIONS:
        cols = [f"{dim}_{m}" for m in GROUPED_DIMENSIONS[dim]]
        raw_vals = [int(user_row[c]) for c in cols]

        # Calculate percentile average
        ranks = [value_norm[c][min(v, len(value_norm[c])-1)] for c, v in zip(cols, raw_vals)]
        
        # Calculate NPS average
        nps_scores = [nps_lookup[c][min(v, len(nps_lookup[c])-1)] for c, v in zip(cols, raw_vals)]

        dim_ranks.append(sum(ranks) / len(ranks) if ranks else 0)
        dim_nps.append(sum(nps_scores) / len(nps_scores) if nps_scores else 0)

    overall_health = sum(dim_ranks) / len(dim_ranks) if dim_ranks else 0

    if overall_health > 0.6: 
        overall_color = COLORS['green']
    elif overall_health > 0.4: 
        overall_color = COLORS['orange']
    else: 
        overall_color = COLORS['red']

    num_vars = len(DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_visible(False)
    ax.set_xticks([])
    ax.set_ylim(0, 1.4)
    ax.grid(False)

    # Draw grid circles
    grid_levels = [0.25, 0.50, 0.75, 1.0]
    for level in grid_levels:
        ax.plot(angles + [angles[0]], [level] * (len(angles)+1),
                color=COLORS['grey'], linestyle=':', linewidth=1, alpha=0.7)

    ax.set_yticks([l * 0.809 for l in grid_levels])
    ax.set_yticklabels([f"{l*100:.0f}%" for l in grid_levels], color="grey", size=10)

    # Draw segments connecting the dimensions
    N = len(DIMENSIONS)
    for i in range(N):
        val = dim_nps[i]
        line_color = COLORS['green'] if val >= 0.5 else (COLORS['red'] if val <= -0.5 else COLORS['orange'])

        prev_i = (i - 1) % N
        next_i = (i + 1) % N

        theta_c, r_c = angles[i], dim_ranks[i]
        theta_p, r_p = angles[prev_i], dim_ranks[prev_i]
        theta_n, r_n = angles[next_i], dim_ranks[next_i]

        t_start_prev = theta_p - 2*np.pi if prev_i > i else theta_p
        t_end_next = theta_n + 2*np.pi if next_i < i else theta_n

        draw_gradient_segment(ax, t_start_prev, r_p, theta_c, r_c, line_color, 0.0, 1.0)
        draw_gradient_segment(ax, theta_c, r_c, t_end_next, r_n, line_color, 1.0, 0.0)

    # Place Icons and Text
    for i, dim in enumerate(DIMENSIONS):
        if dim in loaded_icons:
            img = loaded_icons[dim].copy()
            # Fade effect for the radar icons: modify alpha channel if RGBA
            if img.shape[2] == 4: 
                img[:, :, 3] *= 1
            
            imagebox = OffsetImage(img, zoom=1.15) # Zoom adjusted for Streamlit fit
            # Place icon slightly outside the max radius
            ab = AnnotationBbox(imagebox, (angles[i], 1.25), frameon=False)
            ax.add_artist(ab)

        val = dim_nps[i]
        txt_color = COLORS['green'] if val >= 0.5 else (COLORS['red'] if val <= -0.5 else COLORS['orange'])

        ax.text(angles[i], dim_ranks[i] + 0.15, f"{dim_ranks[i]*100:.0f}%",
                ha='center', va='center', fontweight='bold', fontsize=25, color=txt_color,
                path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    # Center Text
    ax.text(np.pi, 1.3, f"Healthier than {overall_health*100:.0f}% of users",
            ha='center', va='center', fontsize=25, fontweight='bold', color=overall_color,
            bbox=dict(boxstyle="round,pad=0.8", fc="white", ec=overall_color, lw=2))

    plt.tight_layout()
    return fig

# --- 4b. Existing Grid Histogram Logic ---
def draw_histogram_on_ax(ax, full_col_name, user_value, value_freq, nps_lookup):
    freqs = value_freq[full_col_name]
    
    lookup_len = len(nps_lookup[full_col_name])
    safe_idx = min(user_value, lookup_len - 1)
    
    nps_score = nps_lookup[full_col_name][safe_idx]

    status_map = {
        1:  {'color': '#2ecc71', 'label': 'Millor que la resta'},
        0:  {'color': '#f39c12', 'label': 'Igual que la resta'},
        -1: {'color': '#e74c3c', 'label': 'Pitjor que la resta'}
    }
    status = status_map.get(nps_score, {'color': 'gray', 'label': 'Unknown'})

    bar_colors = ['#ecf0f1'] * len(freqs)
    # Highlight current user selection
    if user_value < len(bar_colors):
        bar_colors[user_value] = status['color']

    ax.bar(range(len(freqs)), freqs, color=bar_colors, edgecolor='none')

    ax.grid(False)
    ax.axis('off')

    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('gray')

    fmt_str = ANSWERS_DISPLAY_MAP.get(full_col_name, "{}")
    
    # Format with the numerical user value
    display_text = fmt_str.format(user_value)
    
    # Combine with status label for the final title
    title_str = f"{display_text}\n{status['label']}"

    ax.text(0.5, -0.15, title_str, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=status['color'],
            ha='center', va='top')

def create_survey_grid_figure(user_row, value_norm, value_freq, nps_lookup, loaded_icons):
    
    fig, axes = plt.subplots(5, 5, figsize=(15, 14))

    for col_idx, (dim, metrics) in enumerate(GROUPED_DIMENSIONS.items()):
        dim_ranks = []

        # Plot the 4 metrics (rows 0-3)
        for row_idx_plot, metric in enumerate(metrics):
            ax = axes[row_idx_plot, col_idx]
            full_col = f"{dim}_{metric}"
            
            # Ensure value is integer
            val = int(user_row[full_col])

            draw_histogram_on_ax(ax, full_col, val, value_freq, nps_lookup)

            norm_arr = value_norm[full_col]
            safe_idx = min(val, len(norm_arr) - 1)
            dim_ranks.append(norm_arr[safe_idx])

        # Plot the Icon (row 4)
        ax_icon = axes[4, col_idx]
        avg_ratio = sum(dim_ranks) / len(dim_ranks) if dim_ranks else 0

        if dim in loaded_icons:
            icon_img = get_partial_grayscale_icon(loaded_icons[dim], avg_ratio)
            ax_icon.imshow(icon_img)

        # --- TRANSLATION APPLIED HERE ---
        display_dim = DIMENSION_TRANSLATIONS.get(dim, dim)
        
        ax_icon.axis('off')
        ax_icon.set_title(f"{display_dim}\n{avg_ratio*100:.0f}%",
                          y=-0.25, 
                          fontsize=14, 
                          fontweight='bold',
                          verticalalignment='top')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.6, wspace=0.3, bottom=0.08)
    return fig

def reset_simulation():
    for dim, metrics in GROUPED_DIMENSIONS.items():
        for metric in metrics:
            full_col_name = f"{dim}_{metric}"
            key = f"sim_select_{full_col_name}"
            if key in st.session_state:
                del st.session_state[key]

# --- 5. General Dashboard Render Function ---
def render_dashboard(user_row, value_norm, value_freq, nps_lookup, loaded_icons):
    """
    Combines the Radar Chart and the Grid Figure into a coherent display.
    """
    # 1. Generate Radar Figure
    fig_radar = create_radar_figure(user_row, value_norm, nps_lookup, loaded_icons)
    
    # 2. Generate Grid Figure
    fig_grid = create_survey_grid_figure(user_row, value_norm, value_freq, nps_lookup, loaded_icons)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("#### Puntuació global de salut")
        st.pyplot(fig_radar)
        plt.close(fig_radar)
    
    st.markdown("#### Detall per Dimensió")
    st.pyplot(fig_grid)
    plt.close(fig_grid)

# --- 6. Main Execution ---
def main():
    
    with st.spinner("Loading data..."):
        df_int, nps_lookup, value_norm, value_freq, values_list = load_and_process_data()
        loaded_icons = load_icons()

    if df_int is None:
        return

    all_ids = sorted(df_int.index.tolist())

    if 'selected_user_id' not in st.session_state:
        st.session_state.selected_user_id = all_ids[0]

    current_id_for_defaults = st.session_state.selected_user_id
    default_row_data = df_int.loc[current_id_for_defaults]

    # ==========================================
    # index
    # ==========================================
    st.set_page_config(
        page_title="Qüestionari Estil de Vida",
        page_icon="📊",
        layout="centered"
    )

    st.header("1. Introducció a la pràctica")

    # Student Metadata
    st.markdown("""
    **Assignatura:** M2.959 - Visualització de dades  
    **Estudiant:** Rafael da Silva  
    **Projecte:** Qüestionari interactiu sobre l'estil de vida i el benestar
    """)

    # --- Section 1 ---
    st.subheader("Origen de les dades i context")
    st.markdown("""
    Aquest projecte neix de la necessitat d'anar més enllà de la representació estàtica en un dashboard i crear una eina que realment comuniqui una història a partir de les dades. El conjunt de dades utilitzat és el *Lifestyle and Wellbeing Data*, obtingut originalment a Kaggle.

    * **Font original:** Enquesta realitzada pel portal 360livingmovement.org.
    * **Volum de dades:** El dataset conté 15.972 registres i 24 variables.
    * **Període temporal:** Les dades van ser recollides entre juliol de 2015 i març de 2021,.
    * **Propòsit original:** L'enquesta cercava calcular un "Work-Life Balance Score" per mesurar l'equilibri entre la vida personal i professional.
    """)

    # --- Section 2 ---
    st.subheader("Perspectiva crítica i filosofia de disseny")
    st.markdown("""
    Tot i que el questionari es presenta com una eina de creixement personal, tant el questionari com la resta de la pàgina d'on vé están fortament vinculats a la productivitat empresarial i a la superficialitat comercial.

    Per això he generat les icones per representar com veig realment els pilars originals, que és sota una mirada més cínica:

    * **Cos:** L'estètica del *pretty privilege*.
    * **Ment:** Capacitat resolutiva i eficiència.
    * **Expertesa:** Pur rendiment i productivitat.
    * **Connexió:** Xarxa social orientada a l'ascens laboral.
    * **Propòsit:** Èxit mesurat en acumulació econòmica.

    **Estil Visual: "Corporate Memphis"**
    Per ser coherent amb aquest to superficial i de "perfecció buida", he triat l'estètica *Flat 2.0* o *Corporate Memphis*. Es tracta d'un disseny professional orientat a públics diversos que utilitza colors vibrants i personatges sense trets realistes per transmetre una falsa sensació d'harmonia universal.
    """)

    # --- Section 3 ---
    st.subheader("Funcionament de la visualització interactiva")
    st.markdown("""
    Aquest projecte s'ha dissenyat com un qüestionari interactiu per afavorir l'exploració i la comprensió activa de les dades.

    1.  **Benchmarking en temps real:** L'usuari visualitza immediatament la seva posició relativa respecte als altres 15.972 enquestats del món real. El sistema també genera una puntuació per a cadascun dels 5 pilars i un "Global Wellbeing Score". 
    3.  **Objectiu:** Fomentar una opinió crítica sobre la qualitat de les preguntes i les limitacions del model de felicitat que ens venen les plataformes digitals.
    """)

    # --- ID SELECTOR ---
    col_sel1, col_sel2 = st.columns([1, 4])
    with col_sel1:
        if st.button(f"🔀 {st.session_state.selected_user_id}"):
            st.session_state.selected_user_id = random.choice(all_ids)
            st.rerun()
            
    with col_sel2:
         st.markdown(" <- Clica-hi per feure un altre resposta al atzar.")

    # --- PLOT 2: DATABASE RESULTS ---
    # Retrieve the actual immutable row from the dataframe
    db_row = df_int.loc[st.session_state.selected_user_id]

    # USE THE NEW GENERAL FUNCTION HERE
    render_dashboard(db_row, value_norm, value_freq, nps_lookup, loaded_icons)

    # ==========================================
    st.divider()

    # --- Section 4 (User Interface) ---
    st.header("2. Benvinguda (Interfície d'usuari)")

    # Using a blockquote for the opening quote
    st.markdown("> *\"Coneixe’ns a nosaltres mateixos és el primer pas per a una vida millor.\"*")

    st.markdown("""
    En el món accelerat d'avui, ser conscients de la nostra salut i dels nostres hàbits ens permet prendre decisions informades per viure una vida més harmoniosa i plena. Et convidem a explorar el teu benestar a través d'aquesta experiència interactiva basada en dades globals.

    ### Estructura de l'avaluació
    L'enquesta s'organitza al voltant dels 5 pilars de la salut integral:

    * **Salut Física:** Els teus hàbits diaris i vitalitat corporal.
    * **Salut Mental:** El teu nivell de flow i equilibri emocional.
    * **Productivitat i Expertesa:** La teva capacitat d'assolir reptes.
    * **Connexió Social:** La qualitat del teu entorn i suport humà.
    * **Visió i Propòsit:** La claredat sobre el teu futur i valors.

    ⚠️ **Nota:** És fonamental recordar que, tot i que dividim l'estudi en pilars per facilitar-ne l'anàlisi, el benestar és holístic. No existeix una frontera real entre la ment i el cos, ni entre l'individu i la societat que l'envolta.

    Podeu trobar més informació sobre com treballar aquests pilars a la web oficial del moviment: [360livingmovement.org](http://360livingmovement.org)
    """)
    
    col_header, col_reset = st.columns([5, 1])
    with col_header:
        st.markdown("### Questionari")
    with col_reset:
        st.button("🔄 Esborra", on_click=reset_simulation, type="primary", help="Esborra les teves respostes.")

    user_custom_answers = {}
    
    total_questions = sum(len(metrics) for metrics in GROUPED_DIMENSIONS.values())
    questions_answered = 0


    # --- CUSTOM CSS FOR VERTICAL LINES ---
    st.markdown("""
    <style>
    [data-testid="column"] {
        border-right: 1px solid #e0e0e0;
        padding-right: 1.5rem;
    }
    [data-testid="column"]:last-child {
        border-right: none;
    }
    </style>
    """, unsafe_allow_html=True)

    grid_cols = st.columns(5)

    ICON_FILES_MAP = {
        'Body':       'icon_body.png',
        'Mind':       'icon_mind.png',
        'Expertise':  'icon_expertise.png',
        'Connection': 'icon_connection.png',
        'Purpose':    'icon_purpose.png'
    }

    for i, (dim, metrics) in enumerate(GROUPED_DIMENSIONS.items()):
        with grid_cols[i]:
            # --- 1. DISPLAY CENTERED ICON AND TITLE ---
            icon_filename = ICON_FILES_MAP.get(dim)
            if icon_filename:
                icon_path = os.path.join(ICON_DIR, icon_filename)
                if os.path.exists(icon_path):
                    left_space, center_img, right_space = st.columns([1, 1, 1])
                    with center_img:
                        st.image(icon_path, width=60)
            
            # --- TRANSLATION APPLIED HERE ---
            display_dim = DIMENSION_TRANSLATIONS.get(dim, dim)

            # Centered Title
            st.markdown(f"<h4 style='text-align: center;'>{display_dim}</h4>", unsafe_allow_html=True) 
            
            # --- 2. DISPLAY SLIDERS ---
            for metric in metrics:
                # ... (rest of the slider logic remains exactly the same) ...
                full_col_name = f"{dim}_{metric}"
                
                # Get Question Text
                q_text = QUESTIONS_DICT.get(full_col_name, metric)

                # Get Options
                allowed_values = DISCRETE_VALUES_MAP.get(full_col_name, [])
                if not allowed_values:
                    if df_int is not None and full_col_name in df_int.columns:
                        min_v = int(df_int[full_col_name].min())
                        max_v = int(df_int[full_col_name].max())
                        allowed_values = list(range(min_v, max_v + 1))
                    else:
                        allowed_values = [0, 10]

                # Emoji + Placeholder logic
                emoji_icon = EMOJI_MAP.get(full_col_name, "⚪")
                placeholder_text = PLACEHOLDER_MAP.get(full_col_name, "Select...")
                current_default_option = f"{emoji_icon} {placeholder_text}"

                options_list = [current_default_option] + allowed_values

                # Render Slider
                selected_val = st.select_slider(
                    label=q_text,
                    options=options_list,
                    value=current_default_option,
                    key=f"sim_select_{full_col_name}"
                )
                
                # Track answers
                if selected_val != current_default_option:
                    user_custom_answers[full_col_name] = selected_val
                    questions_answered += 1
                
                st.write("")

    # --- PROGRESS & CONDITIONAL PLOTTING ---

    if questions_answered == total_questions:
        # --- PLOT 1: SIMULATED RESULTS (Only shows when complete) ---
        st.success("Totes les respostes completades! Generant la visualització...")
        st.markdown("#### Resultats de la Simulació")
        
        # USE THE NEW GENERAL FUNCTION HERE
        render_dashboard(user_custom_answers, value_norm, value_freq, nps_lookup, loaded_icons)

    else:
        # --- PLOT 1: EMPTY STATE ---
        st.info("Contesta a les preguntes anteriors per obtenir el teu percentatge de salut.")

    # ==========================================
    # DATABASE VIEWER
    # ==========================================
    st.divider()
    st.header("3. Insights finals")

    st.subheader("La lògica NPS")
    st.markdown("""  
    He tret els colors d'una NPS (Net Promoter Score) estadística. Una resposta individual com ara "dormir 6 hores" és irrellevant de forma aïllada; només guanya sentit quan es posa en relació amb el conjunt de la mostra. Hem convertit aquesta mètrica abstracta en una etiqueta informativa (millor, pitjor o en la mitjana) que permet identificar patrons i outliers de forma immediata.
    """)

    st.subheader("La normalització de dades")
    st.markdown("""  
    I he tret els pencetatges de les visualitzacións d'una normalització per percentils de les respostes. Això em permet fer la mitja de les respostes de cada pilar, i també fer una mitja global. El percentatge de cada pilar m'ha permés fer una comparació diirecta entre dimensions heterogènies dins del gràfic de radar, assegurant que la visualització sigui funcional i autoexplicativa.
    """)

    st.subheader("La unió dels dos per estudiar les demografies: l'edat i el gènere")
    st.markdown("""  
    Però fer una linea de colors gradients dins del radar m'ha permés una cosa encara més útil: veure varies respostes a la vegada.
    """)

    # List of files in the "radars" folder
    files = [
        "radars/wellbeing_study_N10.png",
        "radars/wellbeing_study_N63.png",
        "radars/wellbeing_study_N399.png",
        "radars/wellbeing_study_N2526.png",
        "radars/wellbeing_study_N15971.png"
    ]

    # Create 5 columns in a single row
    cols = st.columns(5)

    # Display each image in its respective column
    for i, file_path in enumerate(files):
        cols[i].image(file_path, width='stretch')

    st.markdown("""  
    Dividint les meves dades per edat i gènere tinc combinacions de mínim 750 respostes. I he trobat que amb N=300 tinc un radars prou diferents entre les demografies. Això que em permet dir coses del conjunt de dades i no només de respostes individuals.
    """)

    st.image("wellbeing_grid_animation.gif", width='stretch')

    st.markdown("""
    
    Podem veure com els pentagons son més verds amb l'edat i més regulars per les dones, els homes tenen la base més petita. Recordant que les columnes són 'Cos', 'Ment', 'Expertesa', 'Connexió' i 'Propòsit', podem fer els insights següents:

    * Amb l'edat la gent té una visió més holística del benestar. Fan esforços en tots els àmbits de manera més equilibrada. 
    * Els homes reporten menys conexió que les dones. 
    * Les dones reporten poca expertesa amb 21-35 anys, però els homes ho fan a totes les edats menys als 51 or more.  

    """)

if __name__ == "__main__":
    main()