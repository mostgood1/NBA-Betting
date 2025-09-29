/*
  NBA Cards Renderer (NHL-style)
  - Loads schedule JSON (data/processed/schedule_2025_26.json)
  - Optionally loads predictions_{YYYY-MM-DD}.csv placed in the repo root
  - Renders game cards by date with team badges and recommendation chips
*/

const state = {
  teams: {}, // tricode -> {name, primary, secondary}
  schedule: [],
  byDate: new Map(),
  scheduleDates: [], // strictly from official schedule
  predsByKey: new Map(), // key: date|homeTricode|awayTricode
  propsEdges: [],
  reconByKey: new Map(), // key: date|homeTricode|awayTricode -> actuals/errors
  reconProps: [], // per player optional
  oddsByKey: new Map(), // date|home|away -> odds snapshot
  propsFilters: { minEdge: 0.03, minEV: 0, stats: new Set(), books: new Set() },
};

// Pin a historical date so it's easy to view in the UI
const PIN_DATE = '2025-04-13';

// Common name aliases that differ from our teams file names
const TEAM_ALIASES = {
  'los angeles clippers': 'LAC',
};

// Optional: treat only official schedule dates as selectable
const STRICT_SCHEDULE_DATES = true;

function fmtLocalTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function fmtLocalDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString();
}

function getQueryParam(name){
  try{
    const u = new URL(window.location.href);
    const v = u.searchParams.get(name);
    return v ? v.trim() : null;
  }catch(e){ return null; }
}

function svgBadgeDataUrl(tri){
  const t = state.teams[tri] || {};
  const bg = (t.primary || '#444').replace('#','%23');
  const fg = '%23FFFFFF';
  const text = encodeURIComponent((tri || '').toUpperCase());
  const name = encodeURIComponent(t.name || tri);
  const svg = `<?xml version="1.0" encoding="UTF-8"?>\
<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96" aria-label="${name}">\
  <rect rx="12" ry="12" width="96" height="96" fill="${bg}"/>\
  <text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" font-family="Inter,Arial,Helvetica,sans-serif" font-weight="700" font-size="42" fill="${fg}">${text}</text>\
</svg>`;
  return `data:image/svg+xml;utf8,${svg}`;
}

function teamLogoUrlByTricode(tri){
  // Prefer local actual logo if provided by the user in assets/logos/ (SVG recommended)
  return `/web/assets/logos/${tri.toUpperCase()}.svg`;
}

function teamLineHTML(tri){
  const t = state.teams[tri] || { name: tri };
  const logo = teamLogoUrlByTricode(tri);
  const fallback = svgBadgeDataUrl(tri);
  const id = (state.teams[tri] && state.teams[tri].id) ? state.teams[tri].id : null;
  const cdn = id ? [
    `https://cdn.nba.com/logos/nba/${id}/primary/L/logo.svg`,
    `https://cdn.nba.com/logos/nba/${id}/primary/L/logo.png`,
    `https://cdn.nba.com/logos/nba/${id}/global/L/logo.svg`,
    `https://cdn.nba.com/logos/nba/${id}/global/L/logo.png`
  ] : [];
  const dataCdn = cdn.join('|');
  // Use local asset first; on error, fallback to png then CDN entries then badge
  const img = `<img class="team-logo" src="${logo}" alt="${t.name} logo" data-tri="${tri}" data-i="-1" data-cdn="${dataCdn}" onerror="handleLogoError(this)" />`;
  return `${img}<div class="name">${t.name}</div>`;
}

function teamPill(team) {
  const t = state.teams[team] || {};
  const bg = t.primary || '#444';
  const text = '#fff';
  const logo = teamLogoUrlByTricode(team);
  const fallback = svgBadgeDataUrl(team);
  const id = (state.teams[team] && state.teams[team].id) ? state.teams[team].id : null;
  const cdn = id ? [
    `https://cdn.nba.com/logos/nba/${id}/primary/L/logo.svg`,
    `https://cdn.nba.com/logos/nba/${id}/primary/L/logo.png`,
    `https://cdn.nba.com/logos/nba/${id}/global/L/logo.svg`,
    `https://cdn.nba.com/logos/nba/${id}/global/L/logo.png`
  ] : [];
  // Use a safe handler and data attributes to avoid HTML attribute quoting issues
  const dataCdn = cdn.join('|');
  const img = `<img class="team-logo" src="${logo}" alt="${t.name || team} logo" data-tri="${team}" data-i="-1" data-cdn="${dataCdn}" onerror="handleLogoError(this)" />`;
  return `<span class="team-pill" style="background:${bg}20;border:1px solid ${bg}60;color:${text}" title="${t.name || team}">${img}<span class="team-name">${t.name || team}</span></span>`;
}

async function loadTeams() {
  const res = await fetch('/web/assets/teams_nba.json');
  const data = await res.json();
  const map = {};
  for (const t of data) map[t.tricode] = t;
  state.teams = map;
}

async function loadSchedule() {
  // Try dynamic API (auto-builds if missing), then fallback to static file
  let sched = [];
  try {
  const r = await fetch('/api/schedule');
    if (r.ok) {
      sched = await r.json();
    }
  } catch(e) { /* ignore */ }
  if (!Array.isArray(sched) || sched.length === 0) {
  const res = await fetch('/data/processed/schedule_2025_26.json');
    sched = await res.json();
  }
  // Filter out non-NBA exhibition teams that won't have logos/mappings
  const isKnown = (tri)=> !!state.teams[String(tri||'').toUpperCase()];
  const filtered = Array.isArray(sched) ? sched.filter(g => isKnown(g.home_tricode) && isKnown(g.away_tricode)) : [];
  state.schedule = filtered;
  const m = new Map();
  const schedDateSet = new Set();
  for (const g of filtered) {
    let dt = g.date_utc || (g.datetime_utc ? g.datetime_utc.slice(0,10) : null);
    if (typeof dt === 'string' && dt.includes('T')) dt = dt.slice(0,10);
    if (!dt) continue;
    schedDateSet.add(dt);
    if (!m.has(dt)) m.set(dt, []);
    m.get(dt).push(g);
  }
  state.byDate = m;
  state.scheduleDates = Array.from(schedDateSet).sort();
}

// If the pinned date isn't present in the schedule, synthesize a slate from predictions_{date}.csv
async function maybeInjectPinnedDate(dateStr){
  try{
    if (!dateStr) return;
    if (state.byDate.has(dateStr)) return; // already present
    const path = `../predictions_${dateStr}.csv`;
    const res = await fetch(path);
    if (!res.ok) return; // no predictions to seed from
    const text = await res.text();
    const rows = parseCSV(text);
    if (!rows || rows.length < 2) return;
    const headers = rows[0];
    const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
    const seen = new Set();
    const list = [];
    for (let i=1;i<rows.length;i++){
      const r = rows[i];
      const home = r[idx.home_team];
      const away = r[idx.visitor_team];
      if (!home || !away) continue;
      const hTri = tricodeFromName(home);
      const aTri = tricodeFromName(away);
      const key = `${hTri}|${aTri}`;
      if (seen.has(key)) continue;
      seen.add(key);
      list.push({
        date_utc: dateStr,
        datetime_utc: `${dateStr}T00:00:00Z`,
        away_tricode: aTri,
        home_tricode: hTri,
        arena_name: '',
        broadcasters_national: '',
      });
    }
    if (list.length){
      state.byDate.set(dateStr, list);
    }
  }catch(e){ /* ignore */ }
}

async function maybeLoadPredictions(dateStr) {
  state.predsByKey.clear();
  const path = `/predictions_${dateStr}.csv`;
  try {
    const res = await fetch(path);
    if (!res.ok) return; // optional
    const text = await res.text();
    const rows = parseCSV(text);
    // Expect columns: date, home_team, visitor_team, home_win_prob, pred_margin, pred_total, ... edges if present
    const headers = rows[0];
    const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
    for (let i=1;i<rows.length;i++){
      const r = rows[i];
      const date = r[idx.date] || r[idx['date']];
      const home = r[idx.home_team];
      const away = r[idx.visitor_team];
      if (!date || !home || !away) continue;
      const key = `${date}|${tricodeFromName(home)}|${tricodeFromName(away)}`;
      state.predsByKey.set(key, Object.fromEntries(headers.map((h,j)=>[h, r[j]])));
    }
  } catch(e) {
    // ignore missing preds
  }
}

async function maybeLoadRecon(dateStr){
  state.reconByKey.clear();
  state.reconProps = [];
  // Games recon
  const gpath = `/data/processed/recon_games_${dateStr}.csv`;
  try{
    const res = await fetch(gpath);
    if (res.ok){
      const text = await res.text();
      const rows = parseCSV(text);
      const headers = rows[0];
      const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
      for (let i=1;i<rows.length;i++){
        const r = rows[i];
        const date = r[idx.date];
        const home = r[idx.home_team];
        const away = r[idx.visitor_team];
        if (!date||!home||!away) continue;
        const key = `${date}|${tricodeFromName(home)}|${tricodeFromName(away)}`;
        const obj = Object.fromEntries(headers.map((h,j)=>[h, r[j]]));
        // normalize numerics
        for (const k of ['home_pts','visitor_pts','actual_margin','total_actual','margin_error','total_error']){
          if (obj[k]!==undefined) obj[k] = Number(obj[k]);
        }
        state.reconByKey.set(key, obj);
      }
    }
  }catch(e){/* ignore */}
  // Props recon (optional)
  const ppath = `/data/processed/recon_props_${dateStr}.csv`;
  try{
    const res = await fetch(ppath);
    if (res.ok){
      const text = await res.text();
      const rows = parseCSV(text);
      const headers = rows[0];
      const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
      const items = [];
      for (let i=1;i<rows.length;i++){
        const r = rows[i];
        const rec = Object.fromEntries(headers.map((h,j)=>[h, r[j]]));
        for (const k of ['pred_pts_err','pred_reb_err','pred_ast_err','pred_threes_err','pred_pra_err']){
          if (rec[k]!==undefined) rec[k] = Number(rec[k]);
        }
        items.push(rec);
      }
      state.reconProps = items;
    }
  }catch(e){/* ignore */}
}

async function maybeLoadPropsEdges(dateStr){
  state.propsEdges = [];
  const path = `/data/processed/props_edges_${dateStr}.csv`;
  try {
    const res = await fetch(path);
    if (!res.ok) return;
    const text = await res.text();
    const rows = parseCSV(text);
    const headers = rows[0];
    const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
    const items = [];
    for (let i=1;i<rows.length;i++){
      const r = rows[i];
      const rec = Object.fromEntries(headers.map((h,j)=>[h, r[j]]));
      // Normalize types
      rec.edge = Number(rec.edge);
      rec.ev = Number(rec.ev);
      items.push(rec);
    }
    state.propsEdges = items;
  } catch(e){ /* ignore */ }
}

async function maybeLoadOdds(dateStr){
  state.oddsByKey.clear();
  // Prefer freshly fetched Bovada game odds over older consensus files
  const candidates = [
    `../data/processed/game_odds_${dateStr}.csv`,
    `../data/processed/closing_lines_${dateStr}.csv`,
    `../data/processed/odds_${dateStr}.csv`,
    `../data/processed/market_${dateStr}.csv`,
  ];
  let text=null;
  for (const p of candidates){
    try{ const r = await fetch(p); if (r.ok){ text = await r.text(); break; } }catch(e){}
  }
  if (!text) return;
  const rows = parseCSV(text);
  if (!rows || rows.length<2) return;
  const headers = rows[0];
  const idx = Object.fromEntries(headers.map((h,i)=>[h,i]));
  const pick = (names)=>{ for (const n of names){ if (idx[n]!==undefined) return n; } return null; };
  const dateCol = pick(['date','game_date','asof_date']);
  const hCol = pick(['home_team','home_name','home','home_tricode']);
  const aCol = pick(['visitor_team','away_team','away_name','away','away_tricode']);
  const hsCol = pick(['home_spread','close_home_spread','spread_home','spread_h','home_line']);
  const asCol = pick(['away_spread','close_away_spread','spread_away','spread_a']);
  const spCol = pick(['spread','close_spread']);
  const totCol = pick(['total','close_total','ou_total','ou_close']);
  const hmlCol = pick(['home_ml','close_home_ml','ml_home']);
  const amlCol = pick(['away_ml','close_away_ml','ml_away']);
  const bookCol = pick(['bookmaker','source','consensus_source']);
  // Price columns to improve EV calc if present
  const hSprPriceCol = pick(['home_spread_price','spread_home_price','home_spread_odds','home_spread_ml']);
  const aSprPriceCol = pick(['away_spread_price','spread_away_price','away_spread_odds','away_spread_ml']);
  const totOverPriceCol = pick(['total_over_price','ou_over_price','total_over_ml','close_total_over_ml']);
  const totUnderPriceCol = pick(['total_under_price','ou_under_price','total_under_ml','close_total_under_ml']);
  for (let i=1;i<rows.length;i++){
    const r = rows[i];
    const d = dateCol ? r[idx[dateCol]].slice(0,10) : dateStr;
    let h = hCol ? r[idx[hCol]] : null; let a = aCol ? r[idx[aCol]] : null;
    if (!h || !a) continue;
    const home = tricodeFromName(h);
    const away = tricodeFromName(a);
    const key = `${d}|${home}|${away}`;
    let home_spread = null, away_spread = null;
    if (hsCol && asCol){
      home_spread = Number(r[idx[hsCol]]);
      away_spread = Number(r[idx[asCol]]);
    } else if (spCol){
      const s = Number(r[idx[spCol]]);
      home_spread = s;
      away_spread = (Number.isFinite(s) ? -s : null);
    }
    // Coerce odds and sanitize zeros (0 means missing/invalid from some feeds)
    const mlH = hmlCol ? Number(r[idx[hmlCol]]) : null;
    const mlA = amlCol ? Number(r[idx[amlCol]]) : null;
    const totV = totCol ? Number(r[idx[totCol]]) : null;
    const hsPrice = hSprPriceCol ? Number(r[idx[hSprPriceCol]]) : null;
    const asPrice = aSprPriceCol ? Number(r[idx[aSprPriceCol]]) : null;
    const toPrice = totOverPriceCol ? Number(r[idx[totOverPriceCol]]) : null;
    const tuPrice = totUnderPriceCol ? Number(r[idx[totUnderPriceCol]]) : null;
    // Basic plausibility filters to drop non–full-game junk that may sneak into CSVs
    const book = bookCol ? String(r[idx[bookCol]] || '').toLowerCase() : '';
    let totClean = (totV===0 ? null : totV);
    if (Number.isFinite(totClean) && totClean < 100) {
      // NBA game totals shouldn't be this low; likely a player/period line -> ignore
      totClean = null;
    }
    let hsClean = Number.isFinite(home_spread) ? Number(home_spread) : null;
    if (hsClean!=null && Math.abs(hsClean) > 60) hsClean = null;
    let asClean = Number.isFinite(away_spread) ? Number(away_spread) : (hsClean!=null? -hsClean : null);
    const rec = {
      home_ml: (mlH===0 ? null : mlH),
      away_ml: (mlA===0 ? null : mlA),
      home_spread: hsClean,
      away_spread: asClean,
      total: totClean,
      bookmaker: bookCol ? r[idx[bookCol]] : null,
      home_spread_price: (hsPrice===0 ? null : hsPrice),
      away_spread_price: (asPrice===0 ? null : asPrice),
      total_over_price: (toPrice===0 ? null : toPrice),
      total_under_price: (tuPrice===0 ? null : tuPrice),
    };
    // Skip if all key fields are missing
    if (rec.home_ml==null && rec.away_ml==null && rec.total==null && rec.home_spread==null) {
      continue;
    }
    state.oddsByKey.set(key, rec);
  }
}

function parseCSV(text) {
  // very basic CSV parser for simple commas without quoted commas
  const lines = text.trim().split(/\r?\n/);
  return lines.map(l => l.split(','));
}

function fmtNum(x, digits=1){
  if (x === null || x === undefined || x==='') return '';
  const n = Number(x);
  if (!Number.isFinite(n)) return '';
  return Math.abs(n) < 1000 ? n.toFixed(digits) : String(Math.round(n));
}

function fmtOddsAmerican(x){
  if (x === null || x === undefined || x==='') return '';
  const n = Number(x);
  if (!Number.isFinite(n)) return '';
  return n > 0 ? `+${Math.round(n)}` : `${Math.round(n)}`;
}

function impliedProbAmerican(odds){
  const o = Number(odds);
  if (!Number.isFinite(o) || o === 0) return null;
  if (o > 0) return 100 / (o + 100);
  return (-o) / ((-o) + 100);
}

function americanToB(odds){
  const o = Number(odds);
  if (!Number.isFinite(o) || o === 0) return null;
  return o > 0 ? (o / 100) : (100 / (-o)); // net decimal (excluding stake)
}

// Approximate standard normal CDF
function normCdf(z){
  // Abramowitz-Stegun approximation
  const t = 1 / (1 + 0.2316419 * Math.abs(z));
  const d = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const p = 1 - d * (0.319381530 * t - 0.356563782 * Math.pow(t,2) + 1.781477937 * Math.pow(t,3) - 1.821255978 * Math.pow(t,4) + 1.330274429 * Math.pow(t,5));
  return z >= 0 ? p : 1 - p;
}

function evFromProbAndAmerican(p, odds){
  const b = americanToB(odds);
  if (b == null || p == null) return null;
  return p * b - (1 - p); // expected return per 1 unit staked
}

function evClass(ev){
  if (ev == null) return '';
  // Align to NHL confidence thresholds
  if (ev >= 0.08) return 'High';
  if (ev >= 0.04) return 'Medium';
  return 'Low';
}

function tricodeFromName(name){
  // quick lookup by team name into teams list
  const lower = String(name).toLowerCase();
  if (TEAM_ALIASES[lower]) return TEAM_ALIASES[lower];
  for (const k in state.teams){
    const t = state.teams[k];
    if (t.name.toLowerCase() === lower) return t.tricode;
  }
  // fallback: already a tricode
  return name.toUpperCase();
}

// Expose a global error handler for logo fallbacks
window.handleLogoError = function(img){
  try{
    const cdn = String(img.dataset.cdn || '').split('|').filter(Boolean);
    let i = Number(img.dataset.i || '-1');
    i += 1;
    img.dataset.i = String(i);
    if (i === 0) {
      img.src = img.src.replace('.svg', '.png');
      return;
    }
    if (i > 0 && i <= cdn.length) {
      img.src = cdn[i-1];
      return;
    }
    // final fallback: generate badge
    const tri = String(img.dataset.tri || '').toUpperCase();
    img.onerror = null;
    img.src = svgBadgeDataUrl(tri);
  } catch(e){
    img.onerror = null;
    const tri = String(img.dataset.tri || '').toUpperCase();
    img.src = svgBadgeDataUrl(tri);
  }
}

function recChips(pred){
  if (!pred) return '';
  const chips = [];
  // Win
  if (pred.home_win_prob) {
    const p = Number(pred.home_win_prob);
    const side = p >= 0.5 ? 'HOME ML' : 'AWAY ML';
    const prob = (p >= 0.5 ? p : 1-p);
    const conf = Math.round(prob*100);
    chips.push(`<span class="chip neutral">${side} ${conf}%</span>`);
  }
  // Spread edge if available
  if (pred.edge_spread) {
    const e = Number(pred.edge_spread);
    const label = e >= 0 ? `HOME ATS +${e.toFixed(1)}` : `AWAY ATS ${e.toFixed(1)}`;
    chips.push(`<span class="chip neutral">${label}</span>`);
  }
  // Total edge
  if (pred.edge_total) {
    const e = Number(pred.edge_total);
    const label = e >= 0 ? `OVER +${e.toFixed(1)}` : `UNDER ${e.toFixed(1)}`;
    chips.push(`<span class="chip neutral">${label}</span>`);
  }
  return chips.join(' ');
}

function resultChips(recon){
  if (!recon) return '';
  const chips = [];
  if (Number.isFinite(recon.home_pts) && Number.isFinite(recon.visitor_pts)){
    chips.push(`<span class="badge badge-final">FINAL ${recon.home_pts}-${recon.visitor_pts}</span>`);
  }
  if (Number.isFinite(recon.margin_error)){
    const e = Number(recon.margin_error);
    const cls = Math.abs(e) <= 2 ? 'good' : (Math.abs(e)<=5 ? 'ok' : 'bad');
    chips.push(`<span class="badge ${cls}" title="Model margin - Actual margin">ΔMargin ${e.toFixed(1)}</span>`);
  }
  if (Number.isFinite(recon.total_error)){
    const e = Number(recon.total_error);
    const cls = Math.abs(e) <= 4 ? 'good' : (Math.abs(e)<=8 ? 'ok' : 'bad');
    chips.push(`<span class="badge ${cls}" title="Model total - Actual total">ΔTotal ${e.toFixed(1)}</span>`);
  }
  return chips.join(' ');
}

function renderDate(dateStr){
  const wrap = document.getElementById('cards');
  if (!wrap) return;
  wrap.innerHTML = '';
  const list = state.byDate.get(dateStr) || [];
  const showResults = document.getElementById('resultsToggle')?.checked;
  const hideOdds = document.getElementById('hideOdds')?.checked;
  // Build simple filters
  const edges = state.propsEdges || [];
  const byTeam = new Map();
  for (const e of edges){
    const t = String(e.team||'').toUpperCase();
    if (!byTeam.has(t)) byTeam.set(t, []);
    byTeam.get(t).push(e);
  }
  for (const g of list){
  const time = g.datetime_est || g.datetime_utc || g.date_est || g.date_utc;
  // Compute local date/time strings preferably from UTC timestamp
  const dtIso = g.datetime_utc || g.datetime_est || null;
  const localTime = dtIso ? fmtLocalTime(dtIso) : (typeof time === 'string' ? (time.includes('T') ? time.split('T')[1].slice(0,5) : '') : fmtLocalTime(time));
  const localDate = dtIso ? fmtLocalDate(dtIso) : (typeof g.date_est === 'string' ? g.date_est.slice(0,10) : (typeof g.date_utc === 'string' ? g.date_utc.slice(0,10) : ''));
  const locBits = [];
  if (g.arena_name) locBits.push(g.arena_name);
  if (g.arena_city) locBits.push(g.arena_city);
  if (g.arena_state) locBits.push(g.arena_state);
  const venueText = locBits.length ? locBits.join(', ') : (g.home_tricode && state.teams[g.home_tricode]?.name ? state.teams[g.home_tricode].name : 'Home');
  const venueLine = `Venue: ${venueText}${g.broadcasters_national?` • TV: ${g.broadcasters_national}`:''} • ${localDate}`;
    const away = g.away_tricode; const home = g.home_tricode;
    const key = `${dateStr}|${home}|${away}`; // note schedule is away@home
    const pred = state.predsByKey.get(key);
  const recs = recChips(pred);
  const recon = showResults ? state.reconByKey.get(key) : null;
  const finals = showResults ? resultChips(recon) : '';
    // Projected / Actual scores
    let projHome=null, projAway=null;
    if (pred && pred.pred_total && pred.pred_margin){
      const T = Number(pred.pred_total), M = Number(pred.pred_margin);
      if (Number.isFinite(T) && Number.isFinite(M)){
        projHome = (T + M) / 2;
        projAway = (T - M) / 2;
      }
    }
  const actualHome = recon && Number.isFinite(recon.home_pts) ? recon.home_pts : null;
  const actualAway = recon && Number.isFinite(recon.visitor_pts) ? recon.visitor_pts : null;
  const totalModel = pred && Number.isFinite(Number(pred.pred_total)) ? Number(pred.pred_total) : null;
  const totalActual = (actualHome!=null && actualAway!=null) ? (actualHome + actualAway) : null;
  const diffLine = (totalModel!=null && totalActual!=null) ? `Diff: ${(totalActual - totalModel).toFixed(2)}` : '';
    const projLine = (projHome!=null && projAway!=null) ? `Projected: ${away} ${fmtNum(projAway,0)} — ${home} ${fmtNum(projHome,0)}` : '';
    const actualLine = (actualHome!=null && actualAway!=null) ? `Final: ${away} ${fmtNum(actualAway,0)} — ${home} ${fmtNum(actualHome,0)}` : '';
    // Props edges badges for the teams
    const tA = String(away||'').toUpperCase();
    const tH = String(home||'').toUpperCase();
    const edgesA = (byTeam.get(tA)||[]).slice().sort((a,b)=>b.edge-a.edge).filter(e=>e.edge>=state.propsFilters.minEdge && e.ev>=state.propsFilters.minEV).slice(0,3);
    const edgesH = (byTeam.get(tH)||[]).slice().sort((a,b)=>b.edge-a.edge).filter(e=>e.edge>=state.propsFilters.minEdge && e.ev>=state.propsFilters.minEV).slice(0,3);
    const badge = (e)=>`<span class="badge" title="${e.stat.toUpperCase()} ${e.side} ${e.line} @ ${e.bookmaker}\nEV ${e.ev.toFixed(2)} | Edge ${(e.edge*100).toFixed(1)}%">${e.player_name}: ${e.stat.toUpperCase()} ${e.side} ${e.line} (${(e.edge*100).toFixed(1)}%)</span>`;
    const propsBadges = [...edgesA.map(badge), ...edgesH.map(badge)].join(' ');

  const odds = state.oddsByKey.get(key);
  const hasAnyOdds = !!(odds && (odds.home_ml!=null || odds.away_ml!=null || Number.isFinite(Number(odds.total)) || Number.isFinite(Number(odds.home_spread))));
    // Compute detailed lines similar to NFL cards
  let oddsBlock = '';
    if (odds){
      const hML = odds.home_ml, aML = odds.away_ml;
      const hImp = impliedProbAmerican(hML); const aImp = impliedProbAmerican(aML);
      const mlLine = `Moneyline (Away / Home) ${fmtOddsAmerican(aML)} / ${fmtOddsAmerican(hML)}`;
      const impLine = (hImp!=null && aImp!=null) ? `Implied Win Prob (Away / Home) ${(aImp*100).toFixed(1)}% / ${(hImp*100).toFixed(1)}%` : '';
      const spr = Number(odds.home_spread);
      const tot = Number(odds.total);
      let spreadLine = '';
      let totalLine = '';
      if (Number.isFinite(spr)){
        const M = pred && Number.isFinite(Number(pred.pred_margin)) ? Number(pred.pred_margin) : null;
        if (M!=null){
          const edge = M - spr; // positive favors Home ATS
          const modelTeam = edge >= 0 ? home : away;
          spreadLine = `Spread (Home) ${fmtNum(spr)} • Model: ${modelTeam} (Edge ${edge>=0?'+':''}${edge.toFixed(2)})`;
        } else {
          spreadLine = `Spread (Home) ${fmtNum(spr)}`;
        }
      }
      if (Number.isFinite(tot)){
        const T = pred && Number.isFinite(Number(pred.pred_total)) ? Number(pred.pred_total) : null;
        if (T!=null){
          const edgeT = T - tot;
          const side = edgeT >= 0 ? 'Over' : 'Under';
          totalLine = `Total ${fmtNum(tot)} • Model: ${side} (Edge ${edgeT>=0?'+':''}${edgeT.toFixed(2)})`;
        } else {
          totalLine = `Total ${fmtNum(tot)}`;
        }
      }
      const parts = [
        `Odds${odds.bookmaker?` @ ${odds.bookmaker}`:''}`,
        mlLine,
        impLine,
        spreadLine,
        totalLine,
      ].filter(Boolean);
      oddsBlock = parts.map(p=>`<div class=\"subtle\">${p}</div>`).join('');
    }
    // Win prob and predicted winner line
    let wpLine = '';
    if (pred && pred.home_win_prob){
      const pHome = Number(pred.home_win_prob);
      if (Number.isFinite(pHome)){
        const pAway = 1 - pHome;
        const winner = pHome >= 0.5 ? home : away;
        wpLine = `Win Prob: Away ${(pAway*100).toFixed(1)}% / Home ${(pHome*100).toFixed(1)}% • Winner: ${winner}`;
      }
    }

    // Accuracy summary (when results available)
    let accuracyLine = '';
    if (showResults && pred && recon && actualHome!=null && actualAway!=null){
      // Winner
      const pHome = Number(pred.home_win_prob);
      const predWinner = pHome >= 0.5 ? home : away;
      const actualWinner = actualHome > actualAway ? home : (actualAway > actualHome ? away : null);
      const winOk = (actualWinner && predWinner === actualWinner);
      // ATS
      let atsOk = null;
      if (odds && Number.isFinite(Number(odds.home_spread))){
        const spr = Number(odds.home_spread);
        const M = Number(pred.pred_margin);
        if (Number.isFinite(M)){
          const predATS = (M - spr >= 0) ? home : away;
          const actualMargin = actualHome - actualAway;
          const actualATS = (actualMargin > spr) ? home : (actualMargin < spr ? away : null);
          atsOk = (actualATS && predATS === actualATS);
        }
      }
      // Totals
      let totOk = null;
      if (odds && Number.isFinite(Number(odds.total))){
        const tot = Number(odds.total);
        const T = Number(pred.pred_total);
        if (Number.isFinite(T)){
          const predSide = (T - tot >= 0) ? 'Over' : 'Under';
          const actualTotal = actualHome + actualAway;
          const actualSide = (actualTotal > tot) ? 'Over' : (actualTotal < tot ? 'Under' : null);
          totOk = (actualSide && predSide === actualSide);
        }
      }
      const tick = (v)=> v===null ? '–' : (v ? '✓' : '✗');
      accuracyLine = `Accuracy: Winner ${tick(winOk)} · ATS ${tick(atsOk)} · Total ${tick(totOk)}`;
    }

    // EV summaries (Winner/Spread/Total) and recommendation candidates
    let evWinnerLine = '';
    let evSpreadLine = '';
    let evTotalLine = '';
    const recCands = [];
    try{
      // Winner EV
      if (odds && pred && (odds.home_ml!=null || odds.away_ml!=null) && pred.home_win_prob!=null){
        const pH = Number(pred.home_win_prob);
        const pA = 1 - pH;
        const evH = evFromProbAndAmerican(pH, odds.home_ml);
        const evA = evFromProbAndAmerican(pA, odds.away_ml);
        let side = null, ev=null;
        if (evH!=null || evA!=null){
          if ((evH??-Infinity) >= (evA??-Infinity)) { side = home; ev = evH; } else { side = away; ev = evA; }
          evWinnerLine = `Winner: ${side} (EV ${(ev*100).toFixed(1)}%) • ${evClass(ev)}`;
        }
        if (evH!=null) recCands.push({market:'moneyline', bet:'home_ml', label:'Home ML', ev:evH, odds:odds.home_ml, book:odds.bookmaker});
        if (evA!=null) recCands.push({market:'moneyline', bet:'away_ml', label:'Away ML', ev:evA, odds:odds.away_ml, book:odds.bookmaker});
      }
      // Spread EV (approximate with normal, sigma assumption)
      if (odds && pred && odds.home_spread!=null && pred.pred_margin!=null){
        const sigmaMargin = 12.0; // rough NBA full-game margin sigma
        const spr = Number(odds.home_spread);
        const M = Number(pred.pred_margin);
        const zHome = (spr - M) / sigmaMargin; // P(Home cover) = 1 - CDF(z)
        const pHomeCover = 1 - normCdf(zHome);
        const pAwayCover = 1 - pHomeCover; // ignoring pushes
        const priceHome = (odds.home_spread_price!=null && odds.home_spread_price!=='') ? Number(odds.home_spread_price) : -110;
        const priceAway = (odds.away_spread_price!=null && odds.away_spread_price!=='') ? Number(odds.away_spread_price) : -110;
        const evH = evFromProbAndAmerican(pHomeCover, priceHome);
        const evA = evFromProbAndAmerican(pAwayCover, priceAway);
        if (evH!=null || evA!=null){
          if ((evH??-Infinity) >= (evA??-Infinity)) { evSpreadLine = `Spread: ${home} (EV ${(evH*100).toFixed(1)}%) • ${evClass(evH)}`; }
          else { evSpreadLine = `Spread: ${away} (EV ${(evA*100).toFixed(1)}%) • ${evClass(evA)}`; }
        }
        if (evH!=null) recCands.push({market:'spread', bet:'home_spread', label:`${home} ATS`, ev:evH, odds:priceHome, book:odds.bookmaker});
        if (evA!=null) recCands.push({market:'spread', bet:'away_spread', label:`${away} ATS`, ev:evA, odds:priceAway, book:odds.bookmaker});
      }
      // Total EV (approximate with normal, sigma assumption)
      if (odds && pred && odds.total!=null && pred.pred_total!=null){
        const sigmaTotal = 20.0; // rough NBA full-game total sigma
        const tot = Number(odds.total);
        const T = Number(pred.pred_total);
        const zOver = (tot - T) / sigmaTotal; // P(Over) = 1 - CDF(z)
        const pOver = 1 - normCdf(zOver);
        const pUnder = 1 - pOver;
        const priceOver = (odds.total_over_price!=null && odds.total_over_price!=='') ? Number(odds.total_over_price) : -110;
        const priceUnder = (odds.total_under_price!=null && odds.total_under_price!=='') ? Number(odds.total_under_price) : -110;
        const evO = evFromProbAndAmerican(pOver, priceOver);
        const evU = evFromProbAndAmerican(pUnder, priceUnder);
        if (evO!=null || evU!=null){
          if ((evO??-Infinity) >= (evU??-Infinity)) { evTotalLine = `Total: Over (EV ${(evO*100).toFixed(1)}%) • ${evClass(evO)}`; }
          else { evTotalLine = `Total: Under (EV ${(evU*100).toFixed(1)}%) • ${evClass(evU)}`; }
        }
         if (evO!=null) recCands.push({market:'totals', bet:'over', label:'Over', ev:evO, odds:priceOver, book:odds.bookmaker});
         if (evU!=null) recCands.push({market:'totals', bet:'under', label:'Under', ev:evU, odds:priceUnder, book:odds.bookmaker});
      }
    }catch(e){ /* ignore EV calc errors */ }
  const tv = g.broadcasters_national || '';
  const venue = venueText;
    // Determine final result class when results are shown
    let w = 0, l = 0, psh = 0;
    const isFinal = showResults && (actualHome!=null && actualAway!=null);
    if (isFinal) {
      // Moneyline
      if (pred && pred.home_win_prob!=null) {
        const pHome = Number(pred.home_win_prob);
        if (Number.isFinite(pHome)){
          const predWinner = pHome >= 0.5 ? home : away;
          const actWinner = actualHome > actualAway ? home : (actualAway > actualHome ? away : null);
          if (actWinner){ if (predWinner === actWinner) w++; else l++; }
        }
      }
      // Totals
      if (odds && Number.isFinite(Number(odds.total))) {
        const tot = Number(odds.total);
        if (totalActual!=null) {
          const actSide = totalActual > tot ? 'Over' : (totalActual < tot ? 'Under' : 'Push');
          if (pred && Number.isFinite(Number(pred.pred_total))){
            const side = (Number(pred.pred_total) - tot >= 0) ? 'Over' : 'Under';
            if (actSide === 'Push') psh++; else if (actSide === side) w++; else l++;
          }
        }
      }
      // ATS
      if (odds && Number.isFinite(Number(odds.home_spread)) && pred && Number.isFinite(Number(pred.pred_margin))) {
        const spr = Number(odds.home_spread);
        const M = Number(pred.pred_margin);
        const predATS = (M - spr >= 0) ? home : away;
        const actualMargin = actualHome - actualAway;
        const actualATS = (actualMargin > spr) ? home : (actualMargin < spr ? away : 'Push');
        if (actualATS === 'Push') psh++; else if (predATS === actualATS) w++; else l++;
      }
    }
    let resultClass = 'final-neutral';
    if (isFinal) {
      if (w > 0 && l === 0) resultClass = 'final-all-win';
      else if (l > 0 && w === 0 && psh === 0) resultClass = 'final-all-loss';
      else if (w > 0 && l > 0) resultClass = 'final-mixed';
      else if (w === 0 && l === 0 && psh > 0) resultClass = 'final-push';
    }
  const node = document.createElement('div');
  node.className = `card ${resultClass}`;
  if (hasAnyOdds) node.setAttribute('data-has-odds','1'); else node.setAttribute('data-has-odds','0');
    // Build detailed card body aligned to NHL example
    let statusLine = '';
    if (showResults && (actualHome!=null && actualAway!=null)) {
      statusLine = 'FINAL';
    } else if (g.game_status_text) {
      statusLine = String(g.game_status_text);
    } else if (localTime) {
      statusLine = `Scheduled ${localTime}`;
    }
    const awayName = state.teams[away]?.name || away;
    const homeName = state.teams[home]?.name || home;
    // Spread and ATS/Totals result if results shown
    let atsLine = '';
    if (odds && Number.isFinite(Number(odds.home_spread))){
      const spr = Number(odds.home_spread);
      const M = pred && Number.isFinite(Number(pred.pred_margin)) ? Number(pred.pred_margin) : null;
      const modelTeam = M!=null ? (M - spr >= 0 ? homeName : awayName) : null;
      let atsResult = '';
      if (showResults && actualHome!=null && actualAway!=null){
        const actualMargin = actualHome - actualAway; // positive => home covers if > spr
        const coversHome = actualMargin > spr || (actualMargin === spr ? null : false);
        const atsTeam = coversHome === null ? 'Push' : (coversHome ? homeName : awayName);
        atsResult = ` • ATS: ${atsTeam}`;
      }
      atsLine = `Spread: ${homeName} ${fmtNum(spr)}${modelTeam?` • Model: ${modelTeam} (Edge ${(M - spr>=0?'+':'')}${(M - spr).toFixed(2)})`:''}${atsResult}`;
    }
    let totalDetailLine = '';
    if (odds && Number.isFinite(Number(odds.total))){
      const tot = Number(odds.total);
      const T = totalModel;
      const side = (T!=null ? (T - tot >= 0 ? 'Over' : 'Under') : null);
      let totResult = '';
      if (showResults && totalActual!=null){
        const r = totalActual > tot ? 'Over' : (totalActual < tot ? 'Under' : 'Push');
        totResult = ` • Totals: ${r}`;
      }
      totalDetailLine = `O/U: ${fmtNum(tot)}${side?` • Model: ${side} (Edge ${(T - tot>=0?'+':'')}${(T - tot).toFixed(2)})`:''}${totResult}`;
    }
    node.setAttribute('data-game-date', dtIso || dateStr);
    node.setAttribute('data-home-abbr', home);
    node.setAttribute('data-away-abbr', away);
  // Build chips: Totals, Spread, and Moneyline
    let chipsTotals = '';
  let chipsSpread = '';
    let chipsMoney = '';
    if (odds) {
      // Totals chips
      const tot = Number(odds.total);
      const T = pred && Number.isFinite(Number(pred.pred_total)) ? Number(pred.pred_total) : null;
      let pOver = null, pUnder = null;
      if (Number.isFinite(tot) && T!=null){
        const sigmaTotal = 20.0;
        const zOver = (tot - T) / sigmaTotal;
        pOver = 1 - normCdf(zOver);
        pUnder = 1 - pOver;
      }
      const priceOver = (odds.total_over_price!=null && odds.total_over_price!=='') ? Number(odds.total_over_price) : null;
      const priceUnder = (odds.total_under_price!=null && odds.total_under_price!=='') ? Number(odds.total_under_price) : null;
      const evO = (pOver!=null && priceOver!=null) ? evFromProbAndAmerican(pOver, priceOver) : null;
      const evU = (pUnder!=null && priceUnder!=null) ? evFromProbAndAmerican(pUnder, priceUnder) : null;
      const clsO = (evO==null? 'neutral' : (evO>0? 'positive' : (evO<0? 'negative' : 'neutral')));
      const clsU = (evU==null? 'neutral' : (evU>0? 'positive' : (evU<0? 'negative' : 'neutral')));
      const evOBadge = (evO==null? '' : `<span class="ev-badge ${evO>0?'pos':(evO<0?'neg':'neu')}">EV ${(evO>0?'+':'')}${(evO*100).toFixed(1)}%</span>`);
      const evUBadge = (evU==null? '' : `<span class="ev-badge ${evU>0?'pos':(evU<0?'neg':'neu')}">EV ${(evU>0?'+':'')}${(evU*100).toFixed(1)}%</span>`);
      const book = (odds.bookmaker || '').toString();
      const bookAbbr = book ? (book.toUpperCase().slice(0,2)) : '';
      const bookBadge = bookAbbr ? `<span class=\"book-badge\" title=\"${book}\">${bookAbbr}</span>` : '';
      const overOddsTxt = (priceOver!=null? fmtOddsAmerican(priceOver) : '—');
      const underOddsTxt = (priceUnder!=null? fmtOddsAmerican(priceUnder) : '—');
      const overProbTxt = (pOver!=null? (pOver*100).toFixed(1)+'%' : '—');
      const underProbTxt = (pUnder!=null? (pUnder*100).toFixed(1)+'%' : '—');
      const isModelOver = (Number.isFinite(tot) && T!=null) ? ((T - tot) >= 0) : false;
      const isModelUnder = (Number.isFinite(tot) && T!=null) ? ((T - tot) < 0) : false;
      const modelBadge = `<span class=\"model-badge\" title=\"Model pick\">PICK</span>`;
      // Push probability (approximate)
      const pushProb = (pOver!=null && pUnder!=null) ? Math.max(0, 1 - pOver - pUnder) : null;
      chipsTotals = `
        <div class=\"row chips\">
          <div class=\"chip title\">Totals ${Number.isFinite(tot)? tot.toFixed(1): ''}</div>
          <div class=\"chip ${clsO} ${isModelOver?'model-pick':''}\">Over ${overOddsTxt} · ${overProbTxt} ${bookBadge} ${evOBadge} ${isModelOver?modelBadge:''}</div>
          <div class=\"chip ${clsU} ${isModelUnder?'model-pick':''}\">Under ${underOddsTxt} · ${underProbTxt} ${bookBadge} ${evUBadge} ${isModelUnder?modelBadge:''}</div>
          ${pushProb!=null ? `<div class=\"chip neutral\">Push · ${(pushProb*100).toFixed(1)}%</div>` : ''}
        </div>`;
      // Spread chips (NBA)
      if (Number.isFinite(Number(odds.home_spread))) {
        const sprH = Number(odds.home_spread);
        const sprA = -sprH;
        const M = pred && Number.isFinite(Number(pred.pred_margin)) ? Number(pred.pred_margin) : null;
        let pHomeCover = null, pAwayCover = null;
        if (M!=null) {
          const sigmaMargin = 12.0;
          const zHome = (sprH - M) / sigmaMargin;
          pHomeCover = 1 - normCdf(zHome);
          pAwayCover = 1 - pHomeCover;
        }
        const priceHome = (odds.home_spread_price!=null && odds.home_spread_price!=='') ? Number(odds.home_spread_price) : null;
        const priceAway = (odds.away_spread_price!=null && odds.away_spread_price!=='') ? Number(odds.away_spread_price) : null;
        const evH = (pHomeCover!=null && priceHome!=null) ? evFromProbAndAmerican(pHomeCover, priceHome) : null;
        const evA = (pAwayCover!=null && priceAway!=null) ? evFromProbAndAmerican(pAwayCover, priceAway) : null;
        const clsH = (evH==null? 'neutral' : (evH>0? 'positive' : (evH<0? 'negative' : 'neutral')));
        const clsA = (evA==null? 'neutral' : (evA>0? 'positive' : (evA<0? 'negative' : 'neutral')));
        const evHBadge = (evH==null? '' : `<span class="ev-badge ${evH>0?'pos':(evH<0?'neg':'neu')}">EV ${(evH>0?'+':'')}${(evH*100).toFixed(1)}%</span>`);
        const evABadge = (evA==null? '' : `<span class="ev-badge ${evA>0?'pos':(evA<0?'neg':'neu')}">EV ${(evA>0?'+':'')}${(evA*100).toFixed(1)}%</span>`);
        const aOddsTxt = (priceAway!=null? fmtOddsAmerican(priceAway): '—');
        const hOddsTxt = (priceHome!=null? fmtOddsAmerican(priceHome): '—');
        const aProbTxt = (pAwayCover!=null? (pAwayCover*100).toFixed(1)+'%': '—');
        const hProbTxt = (pHomeCover!=null? (pHomeCover*100).toFixed(1)+'%': '—');
        const modelHome = (M!=null) ? ((M - sprH) >= 0) : false;
        const modelAway = (M!=null) ? (!modelHome) : false;
        const book = (odds.bookmaker || '').toString();
        const bookAbbr = book ? (book.toUpperCase().slice(0,2)) : '';
        const bookBadge = bookAbbr ? `<span class=\"book-badge\" title=\"${book}\">${bookAbbr}</span>` : '';
        const modelBadge = `<span class=\"model-badge\" title=\"Model pick\">PICK</span>`;
        chipsSpread = `
          <div class=\"row chips\">
            <div class=\"chip title\">Spread</div>
            <div class=\"chip ${clsA} ${modelAway?'model-pick':''}\">Away ${sprA>0?`+${fmtNum(sprA)}`:fmtNum(sprA)} ${aOddsTxt} · ${aProbTxt} ${bookBadge} ${evABadge} ${modelAway?modelBadge:''}</div>
            <div class=\"chip ${clsH} ${modelHome?'model-pick':''}\">Home ${sprH>0?`+${fmtNum(sprH)}`:fmtNum(sprH)} ${hOddsTxt} · ${hProbTxt} ${bookBadge} ${evHBadge} ${modelHome?modelBadge:''}</div>
          </div>`;
      }

      // Moneyline chips
      const hML = odds.home_ml, aML = odds.away_ml;
      const aImp = impliedProbAmerican(aML); const hImp = impliedProbAmerican(hML);
      const pH = pred && pred.home_win_prob!=null ? Number(pred.home_win_prob) : (hImp!=null? hImp : null);
      const pA = (pH!=null) ? (1 - pH) : (aImp!=null? aImp : null);
      const evH = (pH!=null && hML!=null ? evFromProbAndAmerican(pH, hML) : null);
      const evA = (pA!=null && aML!=null ? evFromProbAndAmerican(pA, aML) : null);
      const clsH = (evH==null? 'neutral' : (evH>0? 'positive' : (evH<0? 'negative' : 'neutral')));
      const clsA = (evA==null? 'neutral' : (evA>0? 'positive' : (evA<0? 'negative' : 'neutral')));
      const aOddsTxt = (aML!=null? fmtOddsAmerican(aML): '—');
      const hOddsTxt = (hML!=null? fmtOddsAmerican(hML): '—');
      const aProbTxt = (pA!=null? (pA*100).toFixed(1)+'%': '—');
      const hProbTxt = (pH!=null? (pH*100).toFixed(1)+'%': '—');
      const evABadge = (evA==null? '' : `<span class=\"ev-badge ${evA>0?'pos':(evA<0?'neg':'neu')}\">EV ${(evA>0?'+':'')}${(evA*100).toFixed(1)}%</span>`);
      const evHBadge = (evH==null? '' : `<span class=\"ev-badge ${evH>0?'pos':(evH<0?'neg':'neu')}\">EV ${(evH>0?'+':'')}${(evH*100).toFixed(1)}%</span>`);
      const isModelHome = (pH!=null) ? (pH >= 0.5) : false;
      const isModelAway = (pH!=null) ? (!isModelHome) : false;
      chipsMoney = `
        <div class=\"row chips\">
          <div class=\"chip title\">Moneyline</div>
          <div class=\"chip ${clsA} ${isModelAway?'model-pick':''}\">Away ${aOddsTxt} · ${aProbTxt} ${bookBadge} ${evABadge} ${isModelAway?modelBadge:''}</div>
          <div class=\"chip ${clsH} ${isModelHome?'model-pick':''}\">Home ${hOddsTxt} · ${hProbTxt} ${bookBadge} ${evHBadge} ${isModelHome?modelBadge:''}</div>
        </div>`;
    }

    // Build recommendation and model pick blocks
    let recHtml = '';
    if (recCands.length){
      const best = recCands.slice().sort((a,b)=>((b.ev??-999)-(a.ev??-999)))[0];
      const conf = evClass(best.ev || 0).toLowerCase();
      let recResult = '';
      if (isFinal && odds) {
        try{
          if (best.market==='moneyline'){
            const want = best.bet==='home_ml' ? home : away;
            const act = (actualHome>actualAway)?home:((actualAway>actualHome)?away:null);
            if (act){ recResult = (act===want)?'Win':'Loss'; }
          } else if (best.market==='totals' && Number.isFinite(Number(odds.total))){
            const tot = Number(odds.total);
            const actTotal = totalActual;
            if (actTotal!=null){
              const actSide = actTotal>tot?'Over':(actTotal<tot?'Under':'Push');
              const want = best.bet==='over'?'Over':'Under';
              recResult = (actSide==='Push')?'Push':(actSide===want?'Win':'Loss');
            }
          } else if (best.market==='spread' && Number.isFinite(Number(odds.home_spread))){
            const spr = Number(odds.home_spread);
            const actualMargin = actualHome-actualAway;
            const actATS = (actualMargin>spr)?home:((actualMargin<spr)?away:'Push');
            const want = (best.bet==='home_spread')?home:away;
            recResult = (actATS==='Push')?'Push':(actATS===want?'Win':'Loss');
          }
        }catch(_){/* ignore */}
      }
      const bookAbbr = (best.book||'').toString().toUpperCase().slice(0,2);
      recHtml = `
        <div class=\"row details small\"><div class=\"detail-col\">
          Recommendation: <strong>${best.label}</strong>
          <span class=\"rec-conf ${conf}\">${evClass(best.ev||0)}</span>
          · EV ${(best.ev>0?'+':'')}${(100*(best.ev||0)).toFixed(1)}%
          ${best.odds!=null?` · ${fmtOddsAmerican(best.odds)}`:''}
          ${bookAbbr?` @ ${bookAbbr}`:''}
          ${recResult?` · Result: <strong>${recResult}</strong>`:''}
        </div></div>`;
    }

    let linesLine = '';
    if (odds){
      const hML = (odds.home_ml!=null)?fmtOddsAmerican(odds.home_ml):'—';
      const aML = (odds.away_ml!=null)?fmtOddsAmerican(odds.away_ml):'—';
      const tot = Number.isFinite(Number(odds.total)) ? Number(odds.total).toFixed(1) : '—';
      const spr = Number.isFinite(Number(odds.home_spread)) ? fmtNum(Number(odds.home_spread)) : '—';
      linesLine = `Lines: ML H ${hML} / A ${aML} · Total ${tot} · PL/Spread H ${spr}`;
    }

    let modelPickHtml = '';
    if (pred && pred.home_win_prob!=null){
      const pH = Number(pred.home_win_prob);
      const pickLbl = pH>=0.5 ? 'Home ML' : 'Away ML';
      const pct = (100*Math.max(pH,1-pH)).toFixed(1)+'%';
      modelPickHtml = `<div class=\"row details small\"><div class=\"detail-col\"><div class=\"model-pill\">Model Pick: <strong>${pickLbl}</strong> · ${pct}</div></div></div>`;
    }

    node.innerHTML = `
      <div class="row head">
        <div class="game-date js-local-time">${dtIso || dateStr}</div>
        ${venue ? `<div class=\"venue\">${venue}</div>` : ''}
        <div class="state">${statusLine}</div>
        <div class="period-pill"></div>
        <div class="time-left">${localTime || ''}</div>
        ${isFinal ? `<div class=\"result-badge\">${w}W-${l}L${psh>0?`-${psh}P`:''}</div>` : ''}
      </div>
      <div class="row matchup">
        <div class="team side">
          <div class="team-line">${teamLineHTML(away)}</div>
          <div class="score-block">
            <div class="live-score js-live-away">${(actualAway!=null? fmtNum(actualAway,0) : '—')}</div>
            <div class="sub proj-score">${(projAway!=null? fmtNum(projAway,0) : '—')}</div>
          </div>
        </div>
        <div style="text-align:center; font-weight:700;">@</div>
        <div class="team side">
          <div class="team-line">${teamLineHTML(home)}</div>
          <div class="score-block">
            <div class="live-score js-live-home">${(actualHome!=null? fmtNum(actualHome,0) : '—')}</div>
            <div class="sub proj-score">${(projHome!=null? fmtNum(projHome,0) : '—')}</div>
          </div>
        </div>
      </div>
      ${chipsTotals}
      ${chipsSpread}
      ${chipsMoney}
      ${linesLine?`<div class=\"row details small\"><div class=\"detail-col\"><div>${linesLine}</div></div></div>`:''}
      <div class="row details">
        <div class="detail-col">
          ${totalModel!=null? `<div>Model Total: <strong>${totalModel.toFixed(2)}</strong></div>`: ''}
          ${totalActual!=null? `<div>Actual Total: <strong>${totalActual.toFixed(2)}</strong></div>`: ''}
          ${diffLine? `<div>Diff: <strong>${diffLine.split(': ')[1]}</strong></div>`: ''}
          ${wpLine ? `<div>${wpLine}</div>` : ''}
          ${accuracyLine ? `<div>${accuracyLine}</div>` : ''}
        </div>
      </div>
      ${recHtml}
      ${modelPickHtml}
      ${evWinnerLine ? `<div class=\"row details small\"><div class=\"detail-col\">${evWinnerLine}</div></div>` : ''}
      ${evSpreadLine ? `<div class=\"row details small\"><div class=\"detail-col\">${evSpreadLine}</div></div>` : ''}
      ${evTotalLine ? `<div class=\"row details small\"><div class=\"detail-col\">${evTotalLine}</div></div>` : ''}
      ${atsLine ? `<div class=\"row details small\"><div class=\"detail-col\">${atsLine}</div></div>` : ''}
      ${totalDetailLine ? `<div class=\"row details small\"><div class=\"detail-col\">${totalDetailLine}</div></div>` : ''}
      ${!hideOdds && oddsBlock ? `<div class=\"row details small\"><div class=\"detail-col\">${oddsBlock}</div></div>` : ''}
      ${finals ? `<div class=\"row details small\"><div class=\"detail-col\">${finals}</div></div>` : ''}
      ${propsBadges ? `<div class=\"row details small\"><div class=\"detail-col\">${propsBadges}</div></div>` : ''}
      <div class=\"row details small\"><div class=\"detail-col\"><div>Debug: ${dateStr} | ${home}-${away} | hasOdds=${hasAnyOdds?1:0}</div></div></div>
    `;
    wrap.appendChild(node);
  }
  // Format all game dates into the user's local timezone similar to NHL
  try{
    const nodes = Array.from(document.querySelectorAll('.card .js-local-time'));
    const fmt = new Intl.DateTimeFormat(undefined, { year:'numeric', month:'2-digit', day:'2-digit', hour:'2-digit', minute:'2-digit', hour12:true });
    nodes.forEach(node=>{
      const iso = node.textContent;
      const d = new Date(iso);
      if (!isNaN(d)) node.textContent = fmt.format(d);
    });
  }catch(_){}
}

function setupControls(){
  const picker = document.getElementById('datePicker');
  const applyBtn = document.getElementById('applyBtn');
  const todayBtn = document.getElementById('todayBtn');
  const refreshBtn = document.getElementById('refreshOddsBtn');
  const dates = Array.from(state.byDate.keys()).sort();
  const sched = Array.isArray(state.scheduleDates) ? state.scheduleDates : dates;
  const today = new Date().toISOString().slice(0,10);
  picker.min = dates[0]; picker.max = dates[dates.length-1];
  // Default to the nearest scheduled date to 'today'
  const nearestScheduled = (target)=>{
    const arr = sched;
    if (!arr || arr.length === 0) return null;
    if (arr.includes(target)) return target;
    const t = new Date(target);
    let best = arr[0];
    let bestDiff = Math.abs(new Date(arr[0]) - t);
    for (let i=1;i<arr.length;i++){
      const diff = Math.abs(new Date(arr[i]) - t);
      if (diff < bestDiff){ bestDiff = diff; best = arr[i]; }
    }
    return best;
  };
  const paramDate = getQueryParam('date');
  const defaultDate = (paramDate && (sched.includes(paramDate) ? paramDate : paramDate))
    || nearestScheduled(today)
    || (dates.includes(PIN_DATE) ? PIN_DATE : dates[0]);
  picker.value = defaultDate;
  const go = async ()=>{
    let d = picker.value;
    const hasGames = (state.byDate.get(d) || []).length > 0;
    if (STRICT_SCHEDULE_DATES || !hasGames) {
      const near = nearestScheduled(d);
      if (near && near !== d) {
        d = near; picker.value = near;
      }
    }
    await maybeLoadPredictions(d);
    await maybeLoadOdds(d);
    await maybeLoadPropsEdges(d);
    await maybeLoadRecon(d);
    renderDate(d);
    const note = document.getElementById('note');
    if (note) {
      if ((state.byDate.get(d) || []).length === 0) {
        note.textContent = `No games on ${d}.`;
        note.classList.remove('hidden');
      } else {
        note.classList.add('hidden');
      }
    }
  };
  const apply = ()=> { go(); };
  picker.addEventListener('change', ()=>{}); // wait for Apply to match NHL UX
  if (applyBtn) applyBtn.addEventListener('click', apply);
  const resToggle = document.getElementById('resultsToggle');
  if (resToggle) resToggle.addEventListener('change', go);
  if (refreshBtn) refreshBtn.addEventListener('click', async ()=>{
    const d = picker.value;
    try{
      const url = new URL('/api/cron/refresh-bovada', window.location.origin);
      url.searchParams.set('date', d);
      // If an admin key is configured, the server requires it; allow via prompt or skip
      const key = sessionStorage.getItem('ADMIN_KEY') || '';
      const headers = key ? {'X-Admin-Key': key} : {};
      const resp = await fetch(url.toString(), { method: 'POST', headers });
      if (!resp.ok){ console.warn('Refresh failed', await resp.text()); }
    }catch(e){ console.warn('Refresh error', e); }
    // Re-load odds and re-render
    await maybeLoadOdds(d);
    renderDate(d);
  });
  todayBtn.addEventListener('click', ()=>{
    if (sched.includes(today)) {
      picker.value = today;
    } else {
      const near = (function(){
        const t = new Date(today);
        const arr = sched;
        if (!arr || arr.length === 0) return dates[0];
        let best = arr[0]; let bestDiff = Math.abs(new Date(arr[0]) - t);
        for (let i=1;i<arr.length;i++){
          const diff = Math.abs(new Date(arr[i]) - t);
          if (diff < bestDiff){ bestDiff = diff; best = arr[i]; }
        }
        return best;
      })();
      picker.value = near;
    }
    picker.dispatchEvent(new Event('change'));
  });
  go();
}

(async function init(){
  await loadTeams();
  await loadSchedule();
  // Ensure the pinned date is available in the selector by seeding from predictions if needed
  await maybeInjectPinnedDate(PIN_DATE);
  setupControls();
})();
