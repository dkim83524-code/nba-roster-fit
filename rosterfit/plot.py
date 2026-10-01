"""Render a team's roster-fit square with matplotlib.

Layout: the square on the left (player shapes + team outline); on the right the headline
numbers, team corner percentiles, the player table, and one small square per colored player.
Every player is named on the chart and in the table, so identity never rests on color alone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from . import geometry  # noqa: E402
from .config import CORNERS, Config  # noqa: E402
from .team import PLAYOFF_MODE, TeamResult  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "other": "#898781",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "other": "#898781",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}
FONT = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def _fmt_pct(x: float) -> str:
    return "–" if x is None or np.isnan(x) else f"{x:.0f}"


def _frame(ax, theme, order, labels=None, descriptions=None, rings=None, small=False):
    """Square outline, guide rings ((distance, label) pairs), diagonals and (optionally) corner labels."""
    rings = rings if rings is not None else geometry.rings(330, "percentile")
    ax.set_aspect("equal")
    ax.axis("off")
    for p, _ in rings:
        ax.add_patch(MplPolygon(geometry.CORNER_XY * p, closed=True, fill=False,
                                ec=theme["grid"], lw=0.8, zorder=1))
    for x, y in geometry.CORNER_XY:
        ax.plot([0, x], [0, y], color=theme["grid"], lw=0.8, zorder=1, solid_capstyle="butt")
    ax.add_patch(MplPolygon(geometry.CORNER_XY, closed=True, fill=False, ec=theme["axis"],
                            lw=1.0 if small else 1.2, zorder=1))
    if small or labels is None:
        return
    for p, text in rings:
        ax.text(0, p - 0.012, text, ha="center", va="top", fontsize=8,
                color=theme["muted"], zorder=2)
    for i, corner in enumerate(order):
        x, y = geometry.CORNER_XY[i]
        ha = "left" if x < 0 else "right"
        va = "bottom" if y > 0 else "top"
        dy = 0.05 if y > 0 else -0.05
        head, sub = labels[corner], descriptions.get(corner, "")
        if y > 0:
            ax.text(x, y + dy + 0.055, head, ha=ha, va=va, fontsize=13, fontweight="bold", color=theme["ink"])
            ax.text(x, y + dy, sub, ha=ha, va=va, fontsize=9, color=theme["ink2"])
        else:
            ax.text(x, y + dy, head, ha=ha, va=va, fontsize=13, fontweight="bold", color=theme["ink"])
            ax.text(x, y + dy - 0.06, sub, ha=ha, va=va, fontsize=9, color=theme["ink2"])


def _char_width(ax, fontsize: float) -> float:
    """Approximate width of one character in data units for this axes."""
    fig = ax.get_figure()
    width_in = ax.get_position().width * fig.get_figwidth()
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    return 0.0075 * fontsize / (width_in / span)


def _place_labels(candidates, char_w=0.021, h=0.055, dot=0.025):
    """Greedy label placement: for each label try a few spots and keep the first that collides with
    nothing (earlier labels, other players' dots, the drawing's edge)."""
    placed = []
    dots = [(x - dot, y - dot, x + dot, y + dot) for _, (x, y) in candidates]

    def overlaps(b, o):
        return not (b[2] < o[0] or b[0] > o[2] or b[3] < o[1] or b[1] > o[3])

    def box(x, y, w, ha):
        x0 = x if ha == "left" else x - w
        return (x0, y - h / 2, x0 + w, y + h / 2)

    def hits(b):
        if b[0] < -1.17 or b[2] > 1.17 or b[1] < -1.0 or b[3] > 1.0:
            return True
        return (any(overlaps(b, o) for o in placed)
                or any(overlaps(b, o) for j, o in enumerate(dots) if j != current))

    out = []
    for current, (name, (vx, vy)) in enumerate(candidates):
        w = char_w * len(name) + 0.01
        direction = np.array([vx, vy]) / (np.hypot(vx, vy) or 1)
        chosen = None
        for dist in (0.06, 0.12, 0.2, 0.3):
            for turn in (0, 25, -25, 50, -50, 90, -90, 135, -135, 180):  # inward last, for corner points
                a = np.deg2rad(turn)
                d = np.array([direction[0] * np.cos(a) - direction[1] * np.sin(a),
                              direction[0] * np.sin(a) + direction[1] * np.cos(a)])
                lx, ly = vx + d[0] * dist, vy + d[1] * dist
                ha = "left" if d[0] >= 0 else "right"
                b = box(lx, ly, w, ha)
                if not hits(b):
                    chosen = (lx, ly, ha, b)
                    break
            if chosen:
                break
        if chosen is None:  # crowded: take the plain outward spot, nudged back inside the drawing
            lx, ly = vx + direction[0] * 0.06, vy + direction[1] * 0.06
            ha = "left" if direction[0] >= 0 else "right"
            b = box(lx, ly, w, ha)
            lx += max(0.0, -1.17 - b[0]) - max(0.0, b[2] - 1.17)
            ly += max(0.0, -1.0 - b[1]) - max(0.0, b[3] - 1.0)
            chosen = (lx, ly, ha, box(lx, ly, w, ha))
        placed.append(chosen[3])
        out.append((name, (vx, vy), chosen[:3]))
    return out


def render(result: TeamResult, cfg: Config, out_path: str | Path, theme_name: str | None = None) -> Path:
    theme = THEMES[theme_name or cfg.drawing.theme]
    order = cfg.corners.order
    labels = {c: cfg.label(c) for c in CORNERS}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})

    fig = plt.figure(figsize=(16, 10), facecolor=theme["surface"])
    colored, others, colors = _split(result, cfg, theme)

    # -- title ------------------------------------------------------------------------------
    mode = "playoff rotation (top %d)" % cfg.playoffs.rotation_size if result.mode == PLAYOFF_MODE \
        else "regular season"
    fig.text(0.03, 0.955, f"{result.team_name} · {result.season} {mode}", fontsize=21,
             fontweight="bold", color=theme["ink"], va="center")
    fig.text(0.03, 0.918,
             f"Player shapes: {'rank' if result.scale == 'rank' else 'percentile'} among {result.qualified_n} qualified players "
             f"(≥{cfg.qualified.min_gp} GP, ≥{cfg.qualified.min_mpg:g} MPG); shape area ∝ minutes.  "
             f"Depth: team percentile vs {result.pool_size} team-seasons "
             f"({_season_span(result.pool_seasons)}).",
             fontsize=10.5, color=theme["ink2"], va="center")
    if result.synthetic:
        fig.text(0.97, 0.955, "SYNTHETIC DATA · not real players", fontsize=12, fontweight="bold",
                 color=theme["series"][7], ha="right", va="center")

    # -- the square -------------------------------------------------------------------------
    ax = fig.add_axes([0.02, 0.06, 0.54, 0.83], facecolor=theme["surface"])
    draw_square(ax, result, cfg, theme, colored, others, colors)

    # -- headline numbers -------------------------------------------------------------------
    x0 = 0.6
    fig.text(x0, 0.865, "Coverage", fontsize=11, color=theme["ink2"])
    fig.text(x0, 0.80, f"{result.coverage * 100:.0f}%", fontsize=44, fontweight="bold",
             color=theme["ink"], va="center")
    fig.text(x0, 0.748, "of the square filled by the players' shapes", fontsize=9, color=theme["ink2"])

    share = result.overlap / result.sum_areas * 100 if result.sum_areas else 0.0
    tiles = [
        ("Depth", f"{result.depth * 100:.0f}%", "team outline: whole rotation"),
        ("Overlap", f"{result.overlap:.2f} sq", f"{share:.0f}% of all shape area is shared"),
    ]
    red = " · ".join(f"{labels[c]} +{result.redundancy_players[c]:.1f}"
                     for c in cfg.team.capped_corners if c in result.redundancy_players)
    binding = ", ".join(f"{result.cap_binding_share[c] * 100:.0f}%" for c in cfg.team.capped_corners
                        if c in result.cap_binding_share)
    for i, (head, value, sub) in enumerate(tiles):
        tx = x0 + 0.13 + i * 0.13
        fig.text(tx, 0.865, head, fontsize=10, color=theme["ink2"])
        fig.text(tx, 0.81, value, fontsize=22, fontweight="bold", color=theme["ink"], va="center")
        fig.text(tx, 0.768, sub, fontsize=8.5, color=theme["ink2"], wrap=True)
    fig.text(x0, 0.712, "Redundancy above the cap, in players' worth", fontsize=10, color=theme["ink2"])
    fig.text(x0, 0.688, red or "none", fontsize=11, color=theme["ink"], fontweight="bold")
    fig.text(x0, 0.666, f"Cap binds for {binding} of team-seasons ({cfg.team.cap_mode.replace('_', ' ')})",
             fontsize=8.5, color=theme["ink2"])

    # -- team corner percentiles ------------------------------------------------------------
    fig.text(x0, 0.628, f"Depth by corner: team percentile vs {result.pool_size} team-seasons", fontsize=10.5,
             color=theme["ink2"])
    bx = fig.add_axes([x0, 0.505, 0.36, 0.115], facecolor=theme["surface"])
    bx.axis("off")
    bx.set_xlim(0, 1)
    bx.set_ylim(-0.5, len(order) - 0.5)
    for i, corner in enumerate(order):
        y = len(order) - 1 - i
        bx.text(0.0, y, labels[corner], va="center", fontsize=10, color=theme["ink"])
        bx.barh(y, 0.62, left=0.3, height=0.42, color=theme["grid"])
        bx.barh(y, 0.62 * result.pct[corner] / 100, left=0.3, height=0.42, color=theme["ink"])
        bx.text(0.94, y, _fmt_pct(result.pct[corner]), va="center", ha="left", fontsize=10,
                color=theme["ink"])

    # -- player table -----------------------------------------------------------------------
    tx = fig.add_axes([x0, 0.31, 0.38, 0.185], facecolor=theme["surface"])
    rows = list(colored.iterrows())
    _player_table(tx, result, colored, others, colors, theme, cfg)

    def pts(row, scale=None):
        return _points(row, order, scale)

    # -- small multiples --------------------------------------------------------------------
    per_row = 4
    width = 0.064                                   # figure fraction; height keeps the square square
    height = width * fig.get_figwidth() / fig.get_figheight()
    for k, (_, row) in enumerate(rows[:8]):
        r, col = divmod(k, per_row)
        sx = fig.add_axes([x0 + col * (width + 0.03), 0.165 - r * (height + 0.03), width, height],
                          facecolor=theme["surface"])
        _frame(sx, theme, order, small=True, rings=geometry.rings(result.qualified_n, result.scale)[1:2])
        sx.set_xlim(-1.08, 1.08)
        sx.set_ylim(-1.08, 1.08)
        p = pts(row, scale=1.0)
        c = colors[row["PLAYER_ID"]]
        sx.add_patch(MplPolygon(p, closed=True, fc=c, ec="none", alpha=0.22))
        sx.add_patch(MplPolygon(p, closed=True, fill=False, ec=c, lw=1.6, joinstyle="round"))
        sx.set_title(row["PLAYER_NAME"], fontsize=8.5, color=theme["ink"], pad=2)
    if rows:
        fig.text(x0, 0.297, "Each player at full size (not scaled by minutes)", fontsize=9.5,
                 color=theme["ink2"])

    # -- footer -----------------------------------------------------------------------------
    notes = []
    if result.not_drawn:
        notes.append("Not drawn (missing data): " + ", ".join(result.not_drawn[:6]))
    notes.append(f"Data: NBA.com via nba_api; {' and '.join(cfg.impact.active_sources())} for offense/defense.")
    fig.text(0.03, 0.022, "   ".join(notes), fontsize=8.5, color=theme["muted"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.drawing.dpi, facecolor=theme["surface"])
    plt.close(fig)
    return out_path


def _points(row, order, scale=None):
    vals = {c: row[f"{c}_r"] for c in CORNERS}
    return geometry.shape_points(vals, order, row["scale"] if scale is None else scale)


def _split(result: TeamResult, cfg: Config, theme: dict):
    """(colored players, gray 'others', {player id: color}) for a team result."""
    roster = result.roster
    drawn = roster[roster["drawn"]].sort_values("MIN", ascending=False)
    colored = drawn.head(cfg.drawing.max_colored_players)
    others = drawn.iloc[len(colored):]
    colors = assign_colors(list(colored["PLAYER_ID"]), result.color_order, theme["series"])
    return colored, others, colors


def draw_square(ax, result: TeamResult, cfg: Config, theme: dict, colored, others, colors,
                fontsize: float = 10, legend: bool = True) -> None:
    """The square: rings, diagonals, corner labels, player shapes, team outline, name labels."""
    order = cfg.corners.order
    labels = {c: cfg.label(c) for c in CORNERS}
    _frame(ax, theme, order, labels, cfg.corners.descriptions,
           rings=geometry.rings(result.qualified_n, result.scale))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.25, 1.25)

    for _, row in others.iterrows():
        p = _points(row, order)
        ax.add_patch(MplPolygon(p, closed=True, fc=theme["other"], ec="none", alpha=0.07, zorder=2))
        ax.add_patch(MplPolygon(p, closed=True, fill=False, ec=theme["other"], lw=0.8, alpha=0.6, zorder=2))
    for _, row in colored.iloc[::-1].iterrows():
        p = _points(row, order)
        c = colors[row["PLAYER_ID"]]
        ax.add_patch(MplPolygon(p, closed=True, fc=c, ec="none", alpha=cfg.drawing.fill_alpha, zorder=3))
        ax.add_patch(MplPolygon(p, closed=True, fill=False, ec=c, lw=2, joinstyle="round", zorder=4))

    if cfg.drawing.team_outline:
        team_pts = geometry.shape_points(result.radii, order)
        ax.add_patch(MplPolygon(team_pts, closed=True, fill=False, ec=theme["ink"], lw=2.6,
                                joinstyle="round", zorder=6))

    candidates = []
    for _, row in colored.iterrows():
        p = _points(row, order)
        tip = p[np.argmax(np.hypot(p[:, 0], p[:, 1]))]
        candidates.append((row["PLAYER_NAME"], tuple(tip)))
    placed = _place_labels(candidates, char_w=_char_width(ax, fontsize), h=0.055 * fontsize / 10)
    for (name, (vx, vy), (lx, ly, ha)), (_, row) in zip(placed, colored.iterrows()):
        c = colors[row["PLAYER_ID"]]
        ax.plot([vx, lx], [vy, ly], color=theme["ink2"], lw=0.7, zorder=7)
        ax.scatter([vx], [vy], s=46 * fontsize / 10, color=c, edgecolors=theme["surface"], linewidths=2,
                   zorder=8)
        ax.text(lx, ly, name, ha=ha, va="center", fontsize=fontsize, color=theme["ink"], zorder=9,
                bbox=dict(boxstyle="round,pad=0.15", fc=theme["surface"], ec="none", alpha=0.85))
    if not legend:
        return
    ly = -1.205
    if cfg.drawing.team_outline:
        ax.plot([-1.0, -0.9], [ly, ly], color=theme["ink"], lw=2.6)
        ax.text(-0.87, ly, "Team depth (vs team-seasons)", va="center", fontsize=9.5, color=theme["ink"])
    ax.add_patch(MplPolygon([[-0.15, ly - 0.025], [-0.07, ly - 0.025], [-0.07, ly + 0.025], [-0.15, ly + 0.025]],
                            closed=True, fc=theme["series"][0], alpha=0.25, ec=theme["series"][0], lw=1.5))
    ax.text(-0.04, ly, "Player (vs league)", va="center", fontsize=9.5, color=theme["ink"])
    if len(others):
        ax.add_patch(MplPolygon([[0.5, ly - 0.025], [0.58, ly - 0.025], [0.58, ly + 0.025], [0.5, ly + 0.025]],
                                closed=True, fc=theme["other"], alpha=0.15, ec=theme["other"], lw=1))
        ax.text(0.61, ly, f"{len(others)} more (gray)", va="center", fontsize=9.5, color=theme["ink"])


def _player_table(tx, result: TeamResult, colored, others, colors, theme: dict, cfg: Config,
                  fontsize: float = 9.5) -> None:
    """Rows of name, minutes and the four league percentiles, with a color key per player."""
    order = cfg.corners.order
    labels = {c: cfg.label(c) for c in CORNERS}
    roster = result.roster
    tx.axis("off")
    rows = list(colored.iterrows())
    n = len(rows) + (1 if len(others) else 0) + 1
    tx.set_ylim(-0.5, max(n, 10) - 0.5)
    tx.set_xlim(0, 1)
    cols = [("Player", 0.04, "left"), ("MIN", 0.46, "right")] + \
           [(labels[c].split()[0] if len(labels[c]) > 9 else labels[c], 0.58 + 0.11 * i, "right")
            for i, c in enumerate(order)]
    top = max(n, 10) - 1
    for head, x, ha in cols:
        tx.text(x, top, head, ha=ha, va="center", fontsize=fontsize - 0.5, color=theme["muted"], fontweight="bold")
    for k, (_, row) in enumerate(rows):
        y = top - 1 - k
        tx.scatter([0.012], [y], s=40, color=colors[row["PLAYER_ID"]], edgecolors=theme["surface"], linewidths=1.5)
        tx.text(0.04, y, row["PLAYER_NAME"], va="center", fontsize=fontsize, color=theme["ink"])
        tx.text(0.46, y, f"{row['MIN']:,.0f}", va="center", ha="right", fontsize=fontsize, color=theme["ink"])
        for i, c in enumerate(order):
            tx.text(0.58 + 0.11 * i, y, _fmt_pct(row[f"{c}_pct"]), va="center", ha="right",
                    fontsize=fontsize, color=theme["ink"])
    if len(others):
        y = top - 1 - len(rows)
        mins_share = others["MIN"].sum() / roster["MIN"].sum() * 100
        tx.scatter([0.012], [y], s=40, color=theme["other"], edgecolors=theme["surface"], linewidths=1.5)
        tx.text(0.04, y, f"{len(others)} others ({mins_share:.0f}% of minutes)", va="center",
                fontsize=fontsize, color=theme["ink2"])


def assign_colors(players: list[int], order: list[int], palette: list[str]) -> dict[int, str]:
    """Slot k goes to the team's k-th player by regular-season minutes, in every chart, so a
    player keeps one color. Players outside that top group take the slots left free."""
    rank = {p: i for i, p in enumerate(order)}
    slots = {p: rank[p] for p in players if rank.get(p, len(palette)) < len(palette)}
    free = [i for i in range(len(palette)) if i not in slots.values()]
    for p in players:
        if p not in slots:
            slots[p] = free.pop(0)
    return {p: palette[i] for p, i in slots.items()}


def _season_span(seasons: list[str]) -> str:
    if not seasons:
        return "no complete seasons"
    return seasons[0] if len(seasons) == 1 else f"{seasons[0]} to {seasons[-1]}"


def render_compare(a: TeamResult, b: TeamResult, cfg: Config, out_path: str | Path,
                   theme_name: str | None = None) -> Path:
    """Two teams side by side: their squares, the headline numbers between them, and below,
    each team's player table plus a corner-by-corner comparison of team percentiles."""
    theme = THEMES[theme_name or cfg.drawing.theme]
    order = cfg.corners.order
    labels = {c: cfg.label(c) for c in CORNERS}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT})
    fig = plt.figure(figsize=(18, 11), facecolor=theme["surface"])

    mode = (f"playoff rotations (top {cfg.playoffs.rotation_size})" if a.mode == PLAYOFF_MODE
            else "regular season")
    fig.text(0.03, 0.965, f"{a.team_name} vs {b.team_name} · {a.season} {mode}", fontsize=21,
             fontweight="bold", color=theme["ink"], va="center")
    fig.text(0.03, 0.935,
             f"Player shapes: {'rank' if a.scale == 'rank' else 'percentile'} among {a.qualified_n} qualified players; shape area ∝ minutes "
             f"within each team.  Depth: team percentile vs {a.pool_size} team-seasons "
             f"({_season_span(a.pool_seasons)}).",
             fontsize=10.5, color=theme["ink2"], va="center")
    if a.synthetic or b.synthetic:
        fig.text(0.97, 0.965, "SYNTHETIC DATA · not real players", fontsize=12, fontweight="bold",
                 color=theme["series"][7], ha="right", va="center")

    sides = ((a, 0.0, 0.03, 0.03), (b, 0.58, 0.61, 0.61))
    for res, ax_x, head_x, table_x in sides:
        colored, others, colors = _split(res, cfg, theme)
        fig.text(head_x, 0.893, res.team_name, fontsize=16, fontweight="bold", color=theme["ink"])
        ax = fig.add_axes([ax_x, 0.30, 0.42, 0.57], facecolor=theme["surface"])
        draw_square(ax, res, cfg, theme, colored, others, colors, fontsize=9, legend=False)
        tx = fig.add_axes([table_x, 0.025, 0.36, 0.21], facecolor=theme["surface"])
        _player_table(tx, res, colored, others, colors, theme, cfg, fontsize=8.5)

    # -- headline numbers between the squares ---------------------------------------------------
    xa, xb, xm = 0.462, 0.538, 0.5
    fig.text(xa, 0.86, a.team, ha="center", fontsize=12, fontweight="bold", color=theme["ink"])
    fig.text(xb, 0.86, b.team, ha="center", fontsize=12, fontweight="bold", color=theme["ink"])
    fig.text(xm, 0.83, "Coverage (players' shapes)", ha="center", fontsize=10.5, color=theme["ink2"])
    for x, res in ((xa, a), (xb, b)):
        fig.text(x, 0.795, f"{res.coverage * 100:.0f}%", ha="center", va="center", fontsize=26,
                 fontweight="bold", color=theme["ink"])
    rows = [("Depth (team outline)", lambda r: f"{r.depth * 100:.0f}%"),
            ("Overlap (squares)", lambda r: f"{r.overlap:.2f}")]
    rows += [(f"Redundancy: {labels[c]}", lambda r, c=c: f"+{r.redundancy_players.get(c, 0):.1f}")
             for c in cfg.team.capped_corners]
    y = 0.74
    for title, fmt in rows:
        fig.text(xm, y, title, ha="center", fontsize=9, color=theme["ink2"])
        fig.text(xa, y - 0.027, fmt(a), ha="center", fontsize=13, fontweight="bold", color=theme["ink"])
        fig.text(xb, y - 0.027, fmt(b), ha="center", fontsize=13, fontweight="bold", color=theme["ink"])
        y -= 0.075
    fig.text(xm, y + 0.02, "redundancy in players' worth", ha="center", fontsize=8, color=theme["muted"])
    lg = fig.add_axes([0.425, 0.31, 0.15, 0.05], facecolor=theme["surface"])
    lg.axis("off")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    if cfg.drawing.team_outline:
        lg.plot([0.0, 0.12], [0.75, 0.75], color=theme["ink"], lw=2.6)
        lg.text(0.16, 0.75, "Team depth outline (vs team-seasons)", va="center", fontsize=8.5,
                color=theme["ink"])
    lg.add_patch(MplPolygon([[0.0, 0.12], [0.12, 0.12], [0.12, 0.38], [0.0, 0.38]], closed=True,
                            fc=theme["series"][0], alpha=0.25, ec=theme["series"][0], lw=1.2))
    lg.text(0.16, 0.25, "Player shape (vs league)", va="center", fontsize=8.5, color=theme["ink"])

    # -- corner-by-corner comparison --------------------------------------------------------------
    fig.text(xm, 0.272, "Depth by corner (team percentile)", ha="center", fontsize=10.5, color=theme["ink2"])
    fig.text(xm, 0.25, f"● {a.team}    ○ {b.team}", ha="center", fontsize=10, color=theme["ink"])
    dx = fig.add_axes([0.415, 0.03, 0.17, 0.205], facecolor=theme["surface"])
    dx.axis("off")
    dx.set_xlim(-4, 104)
    dx.set_ylim(-0.6, len(order) - 0.3)
    for i, corner in enumerate(order):
        yy = len(order) - 1 - i
        pa, pb = a.pct[corner], b.pct[corner]
        dx.text(0, yy + 0.32, f"{labels[corner]}   {a.team} {_fmt_pct(pa)} · {b.team} {_fmt_pct(pb)}",
                fontsize=8.5, color=theme["ink"], va="center")
        dx.plot([0, 100], [yy, yy], color=theme["grid"], lw=1, solid_capstyle="butt", zorder=1)
        dx.plot([min(pa, pb), max(pa, pb)], [yy, yy], color=theme["ink2"], lw=2, zorder=2)
        dx.scatter([pb], [yy], s=70, facecolors=theme["surface"], edgecolors=theme["ink"], linewidths=1.8,
                   zorder=3)
        dx.scatter([pa], [yy], s=70, color=theme["ink"], edgecolors=theme["surface"], linewidths=1.5,
                   zorder=4)
    for x in (0, 50, 100):
        dx.text(x, -0.5, f"{x}", ha="center", va="center", fontsize=7.5, color=theme["muted"])

    fig.text(0.97, 0.008, f"Data: NBA.com via nba_api; {' and '.join(cfg.impact.active_sources())} for offense/defense.",
             fontsize=8, color=theme["muted"], ha="right")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.drawing.dpi, facecolor=theme["surface"])
    plt.close(fig)
    return out_path
