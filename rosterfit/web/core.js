/* Roster Fit core: the same math as rosterfit/team.py and rosterfit/geometry.py, for the browser.
 * No DOM access here, so Node can load it for the parity test against Python.
 *
 * Square: [-1, 1] x [-1, 1] (area 4); corners clockwise from top-left. A shape puts a vertex on
 * each center-to-corner diagonal at distance radius(percentile) of the way to that corner.
 */
(function (root) {
  "use strict";

  const CORNER_XY = [[-1, 1], [1, 1], [1, -1], [-1, -1]];
  const SQUARE_AREA = 4;

  function cross(ax, ay, bx, by) {
    return ax * by - ay * bx;
  }

  /* Distance from the center (0-1) for a percentile among n players (or team-seasons); same as
   * geometry.radius in Python. "percentile": pct / 100. "rank": 1 - ln(rank) / ln(1 + n), rank = 1 +
   * players ahead: the best reaches the corner and every halving of the rank adds the same step. */
  function radius(pct, n, scale) {
    if (pct == null || Number.isNaN(pct)) return null;
    const p = Math.min(Math.max(pct / 100, 0), 1);
    if (scale !== "rank") return p;
    const m = Math.max(n || 1, 1);
    return 1 - Math.log1p(m * (1 - p)) / Math.log1p(m);
  }

  /* Guide rings as [distance, label]: top 5 / 20 / 50 on the rank scale, 25 / 50 / 75 otherwise. */
  function rings(n, scale) {
    if (scale !== "rank") return [0.25, 0.5, 0.75].map((p) => [p, String(p * 100)]);
    const m = Math.max(n || 1, 1);
    return [5, 20, 50].map((k) => [1 - Math.log(k) / Math.log1p(m), `top ${k}`]);
  }

  function shapePoints(vals, scale) {
    return CORNER_XY.map(([x, y], i) => [x * vals[i] * scale, y * vals[i] * scale]);
  }

  /* Share of the square: only neighboring corners multiply. */
  function shapeArea(vals, scale) {
    let s = 0;
    for (let i = 0; i < 4; i++) s += vals[i] * vals[(i + 1) % 4];
    return (scale * scale * s) / 4;
  }

  /* Distance from the center to a shape's edge along direction (ux, uy). */
  function rayRadius(pts, ux, uy) {
    let best = 0;
    for (let i = 0; i < 4; i++) {
      const a = pts[i];
      const b = pts[(i + 1) % 4];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const den = cross(ux, uy, dx, dy);
      if (Math.abs(den) < 1e-12) continue;
      const t = cross(a[0], a[1], ux, uy) / den;
      if (t < -1e-9 || t > 1 + 1e-9) continue;
      const r = cross(a[0], a[1], dx, dy) / den;
      if (r > best) best = r;
    }
    return best;
  }

  /* Area the shapes cover together, as a share of the square. Every shape is star-shaped around
   * the center, so the union's edge in any direction is the farthest shape edge: integrate it. */
  function unionArea(shapes, samples) {
    const n = samples || 4096;
    if (!shapes.length) return 0;
    let total = 0;
    for (let k = 0; k < n; k++) {
      const th = ((k + 0.5) / n) * 2 * Math.PI;
      const ux = Math.cos(th);
      const uy = Math.sin(th);
      let r = 0;
      for (const pts of shapes) {
        const q = rayRadius(pts, ux, uy);
        if (q > r) r = q;
      }
      total += r * r;
    }
    return (0.5 * total * ((2 * Math.PI) / n)) / SQUARE_AREA;
  }

  /* 100 x share of a sorted pool at or below v (the convention used everywhere). */
  function pctAgainst(sorted, v) {
    if (v == null || Number.isNaN(v) || !sorted || !sorted.length) return null;
    const x = v + 1e-9;
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sorted[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return (100 * lo) / sorted.length;
  }

  /* Score a lineup. entries: [{pid, season, name, min, pct: [4], z: [4]}], corners in canonical order.
   * ctx: {meta, pools, caps, oneWorth, drawAll, fixedWeight, equalSize} */
  function evaluate(entries, ctx) {
    const m = ctx.meta;
    const corners = m.corners;
    const order = m.order;
    const idx = Object.fromEntries(corners.map((c, i) => [c, i]));
    const total = entries.reduce((s, e) => s + e.min, 0);
    const players = entries.map((e) => ({
      ...e,
      weight: ctx.fixedWeight != null ? ctx.fixedWeight : total > 0 ? e.min / (total / m.slots) : 0,
      contrib: {},
    }));
    const values = {};
    const uncapped = {};
    const redundancy = {};
    const redundancyPlayers = {};
    const missing = {};
    for (const c of corners) {
      const i = idx[c];
      let sum = 0;
      let miss = 0;
      for (const p of players) {
        let z = p.z[i];
        if (z == null) miss += p.min;
        if (m.floor.includes(c)) z = z == null ? 0 : Math.max(z, 0);
        else z = z == null ? m.missing_value_z : z;
        p.contrib[c] = p.weight * z;
        sum += p.weight * z;
      }
      uncapped[c] = sum;
      missing[c] = total ? miss / total : 0;
      if (m.capped.includes(c)) {
        const cap = ctx.caps ? ctx.caps[c] : null;
        if (cap == null || Number.isNaN(cap)) {
          values[c] = sum;
          redundancy[c] = 0;
        } else {
          values[c] = Math.min(sum, cap);
          redundancy[c] = Math.max(sum - cap, 0);
        }
        const one = ctx.oneWorth ? ctx.oneWorth[c] : null;
        redundancyPlayers[c] = one > 0 ? redundancy[c] / one : null;
      } else {
        values[c] = sum;
      }
    }
    const pct = {};
    for (const c of corners) pct[c] = pctAgainst(ctx.pools[c], values[c]);
    const sc = m.scale || "percentile";
    const outline = order.map((c) => radius(pct[c], (ctx.pools[c] || []).length, sc) || 0);
    const depth = shapeArea(outline, 1);
    const nOf = (season) => (m.qualified && m.qualified[season]) || 1;

    for (const p of players) {
      const complete = p.pct.every((v) => v != null);
      p.drawn = complete && (ctx.drawAll || p.min >= m.min_minutes_to_draw);
    }
    const drawn = players.filter((p) => p.drawn);
    const ref = drawn.length ? Math.max(...drawn.map((p) => p.min)) : 1;
    const shapes = [];
    let sumAreas = 0;
    for (const p of players) {
      p.scale = ctx.equalSize || m.minutes_scaling !== "area" ? 1 : Math.min(1, Math.sqrt(p.min / ref));
      if (!p.drawn) {
        p.area = null;
        continue;
      }
      p.vals = order.map((c) => radius(p.pct[idx[c]], nOf(p.season), sc));
      p.points = shapePoints(p.vals, p.scale);
      p.area = shapeArea(p.vals, p.scale);
      if (p.area > 1e-12) {
        sumAreas += p.area;
        shapes.push(p.points);
      }
    }
    const coverage = unionArea(shapes);
    return {
      players, values, uncapped, redundancy, redundancyPlayers, missing, pct, outline, depth,
      coverage, sumAreas, overlap: Math.max(sumAreas - coverage, 0),
    };
  }

  /* data.players[season] rows -> {season: Map(pid -> player)}. Row: pid, name, team, games, minutes,
   * qualified, percentile per corner, z per corner, offensive role, defensive role, position. */
  function playerIndex(data) {
    const nC = data.meta.corners.length;
    const out = {};
    for (const s of Object.keys(data.players)) {
      const map = new Map();
      for (const r of data.players[s]) {
        map.set(r[0], {
          pid: r[0], season: s, name: r[1], team: r[2], gp: r[3], min: r[4], qualified: !!r[5],
          pct: r.slice(6, 6 + nC), z: r.slice(6 + nC, 6 + 2 * nC),
          offRole: r[6 + 2 * nC] || null, defRole: r[7 + 2 * nC] || null, pos: r[8 + 2 * nC] || null,
        });
      }
      out[s] = map;
    }
    return out;
  }

  function entriesFor(index, season, pairs) {
    const blank = [null, null, null, null];
    return pairs.map(([pid, min]) => {
      const p = index[season] ? index[season].get(pid) : null;
      return { pid, season, name: p ? p.name : String(pid), min, pct: p ? p.pct : blank, z: p ? p.z : blank,
        offRole: p ? p.offRole : null, defRole: p ? p.defRole : null, pos: p ? p.pos : null };
    });
  }

  function teamContext(data, season, mode) {
    return {
      meta: data.meta,
      pools: data.pools[mode],
      caps: (data.caps[mode] || {})[season] || {},
      oneWorth: data.one_worth[season] || {},
      drawAll: mode === "playoffs",
    };
  }

  function evaluateTeam(data, index, season, abbr, mode) {
    const team = data.teams[season] && data.teams[season][abbr];
    const pairs = team ? team[mode] || [] : [];
    return evaluate(entriesFor(index, season, pairs), teamContext(data, season, mode));
  }

  /* A made-up lineup: every player a full-time starter (weight 1, full-size shape). Empty spots
   * count as league-average players. Caps and depth use the latest season's regular pool. */
  function evaluateLineup(data, index, picks) {
    const latest = data.meta.seasons[data.meta.seasons.length - 1];
    const entries = picks.map(({ season, pid }) => {
      const p = index[season].get(pid);
      return { pid, season, name: p.name, min: 1, pct: p.pct, z: p.z, offRole: p.offRole, defRole: p.defRole, pos: p.pos };
    });
    const ctx = teamContext(data, latest, "regular");
    return evaluate(entries, { ...ctx, drawAll: true, fixedWeight: 1, equalSize: true });
  }

  /* Slot k goes to the team's k-th player by regular-season minutes in every view, so a player keeps
   * one color; players outside that group take the free slots. */
  function assignColors(players, order, n) {
    const rank = new Map(order.map((p, i) => [p, i]));
    const slots = new Map();
    for (const p of players) if (rank.has(p) && rank.get(p) < n) slots.set(p, rank.get(p));
    const used = new Set(slots.values());
    const free = [];
    for (let i = 0; i < n; i++) if (!used.has(i)) free.push(i);
    for (const p of players) if (!slots.has(p)) slots.set(p, free.shift());
    return slots;
  }

  /* Greedy label placement (same rules as plot.py): try spots around the point, keep the first
   * that stays inside the drawing and clear of earlier labels and of the other players' dots. */
  function placeLabels(cands, charW, h, dot) {
    const r = dot == null ? 0.025 : dot;
    const placed = [];
    const dots = cands.map(({ x, y }) => [x - r, y - r, x + r, y + r]);
    const overlaps = (b, o) => !(b[2] < o[0] || b[0] > o[2] || b[3] < o[1] || b[1] > o[3]);
    const box = (x, y, w, ha) => {
      const x0 = ha === "start" ? x : x - w;
      return [x0, y - h / 2, x0 + w, y + h / 2];
    };
    let current = -1;
    const hits = (b) => {
      if (b[0] < -1.17 || b[2] > 1.17 || b[1] < -1.0 || b[3] > 1.0) return true;
      return placed.some((o) => overlaps(b, o)) || dots.some((o, j) => j !== current && overlaps(b, o));
    };
    const out = [];
    for (const { name, x: vx, y: vy } of cands) {
      current += 1;
      const w = charW * name.length + 0.01;
      const len = Math.hypot(vx, vy) || 1;
      const dir = [vx / len, vy / len];
      let chosen = null;
      for (const dist of [0.06, 0.12, 0.2, 0.3]) {
        for (const turn of [0, 25, -25, 50, -50, 90, -90, 135, -135, 180]) {
          const a = (turn * Math.PI) / 180;
          const d = [dir[0] * Math.cos(a) - dir[1] * Math.sin(a), dir[0] * Math.sin(a) + dir[1] * Math.cos(a)];
          const lx = vx + d[0] * dist;
          const ly = vy + d[1] * dist;
          const ha = d[0] >= 0 ? "start" : "end";
          const b = box(lx, ly, w, ha);
          if (!hits(b)) {
            chosen = { lx, ly, ha, b };
            break;
          }
        }
        if (chosen) break;
      }
      if (!chosen) {
        let lx = vx + dir[0] * 0.06;
        let ly = vy + dir[1] * 0.06;
        const ha = dir[0] >= 0 ? "start" : "end";
        const b = box(lx, ly, w, ha);
        lx += Math.max(0, -1.17 - b[0]) - Math.max(0, b[2] - 1.17);
        ly += Math.max(0, -1.0 - b[1]) - Math.max(0, b[3] - 1.0);
        chosen = { lx, ly, ha, b: box(lx, ly, w, ha) };
      }
      placed.push(chosen.b);
      out.push({ name, x: vx, y: vy, lx: chosen.lx, ly: chosen.ly, ha: chosen.ha });
    }
    return out;
  }

  /* ---- cap game ---- */

  /* Seeded random numbers (mulberry32), so a daily deal comes out the same for everyone. */
  function seededRandom(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* FNV-1a hash of a string, for seeds like "2026-27|2026-10-01". */
  function hashSeed(text) {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  }

  /* The order teams are dealt in: a seeded shuffle of the sorted team list. */
  function dealOrder(teams, seed) {
    const rand = seededRandom(seed);
    const out = teams.slice().sort();
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  }

  /* What to keep in hand for one open spot: the lowest salary that `teams` different teams have
   * someone at or under. With teams = 5, even after four teams are used up, one is left with a
   * player you can afford (the deal only offers teams that have one). Fewer teams than that: the
   * most expensive team's cheapest player. */
  function signingFloor(salariesByTeam, teams) {
    const mins = salariesByTeam.filter((l) => l.length).map((l) => Math.min(...l)).sort((a, b) => a - b);
    if (!mins.length) return 0;
    return mins[Math.min(Math.max(teams, 1), mins.length) - 1];
  }

  /* A signing must leave `reserve` in hand: the floors of the spots still open after it. */
  function canAfford(salary, left, reserve) {
    return salary + reserve <= left + 1e-6;
  }

  const api = {
    CORNER_XY, radius, rings, shapePoints, shapeArea, unionArea, pctAgainst, evaluate, playerIndex, entriesFor,
    teamContext, evaluateTeam, evaluateLineup, assignColors, placeLabels,
    seededRandom, hashSeed, dealOrder, signingFloor, canAfford,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.RosterFitCore = api;
})(typeof window !== "undefined" ? window : globalThis);
