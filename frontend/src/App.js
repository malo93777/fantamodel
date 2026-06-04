import React, { createContext, useContext, useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { Toaster, toast } from "sonner";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, CartesianGrid } from "recharts";
import "./App.css";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;
axios.defaults.withCredentials = true;

// =============== AUTH CONTEXT ===============
const AuthCtx = createContext(null);
export const useAuth = () => useContext(AuthCtx);

function AuthProvider({ children }) {
  const [user, setUser] = useState(null); // null = checking, false = logged out, obj = user
  useEffect(() => {
    axios.get(`${API}/auth/me`).then(r => setUser(r.data)).catch(() => setUser(false));
  }, []);
  const logout = async () => {
    try { await axios.post(`${API}/auth/logout`); } catch (_) {}
    setUser(false);
  };
  return <AuthCtx.Provider value={{ user, setUser, logout }}>{children}</AuthCtx.Provider>;
}

// =============== SEARCHABLE COMBOBOX ===============
function SearchableSelect({ options, value, onChange, placeholder = "Cerca...", testId, getLabel = (o) => o.display, getValue = (o) => o.raw }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const ref = React.useRef(null);

  useEffect(() => {
    if (value && options) {
      const cur = options.find(o => getValue(o) === value);
      if (cur) setQ(getLabel(cur));
    } else if (!value) setQ("");
    // eslint-disable-next-line
  }, [value, options]);

  useEffect(() => {
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const filtered = (options || []).filter(o => getLabel(o).toLowerCase().includes(q.toLowerCase())).slice(0, 100);

  const pick = (o) => { onChange(getValue(o)); setQ(getLabel(o)); setOpen(false); };
  const onKey = (e) => {
    if (!open) setOpen(true);
    if (e.key === "ArrowDown") { e.preventDefault(); setHi(h => Math.min(h + 1, filtered.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setHi(h => Math.max(h - 1, 0)); }
    else if (e.key === "Enter" && filtered[hi]) { e.preventDefault(); pick(filtered[hi]); }
    else if (e.key === "Escape") setOpen(false);
  };

  return (
    <div className="combo" ref={ref}>
      <input className="input" value={q} onChange={e => { setQ(e.target.value); setOpen(true); setHi(0); }}
        onFocus={() => setOpen(true)} onKeyDown={onKey} placeholder={placeholder} data-testid={testId} autoComplete="off" />
      {open && filtered.length > 0 && (
        <div className="combo-list" data-testid={testId ? `${testId}-list` : undefined}>
          {filtered.map((o, i) => (
            <div key={getValue(o)} className={"combo-item" + (i === hi ? " combo-item-hi" : "")}
              onMouseEnter={() => setHi(i)} onMouseDown={(e) => { e.preventDefault(); pick(o); }}>
              {getLabel(o)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SearchableMultiSelect({ options, values, onChange, placeholder = "Cerca e aggiungi giocatori...", testId, getLabel = (o) => o.display, getValue = (o) => o.raw }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const ref = React.useRef(null);

  useEffect(() => {
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const filtered = (options || []).filter(o => !values.includes(getValue(o)) && getLabel(o).toLowerCase().includes(q.toLowerCase())).slice(0, 100);

  return (
    <div className="combo" ref={ref}>
      <div className="chips-input">
        {values.map(v => {
          const o = options?.find(x => getValue(x) === v);
          return (
            <span key={v} className="chip" data-testid={`chip-${v}`}>
              {o ? getLabel(o) : v}
              <button onClick={() => onChange(values.filter(x => x !== v))} className="chip-x" aria-label="rimuovi">×</button>
            </span>
          );
        })}
        <input className="chips-search" value={q} onChange={e => { setQ(e.target.value); setOpen(true); }}
          onFocus={() => setOpen(true)} placeholder={values.length ? "" : placeholder} data-testid={testId} autoComplete="off" />
      </div>
      {open && filtered.length > 0 && (
        <div className="combo-list" data-testid={testId ? `${testId}-list` : undefined}>
          {filtered.map(o => (
            <div key={getValue(o)} className="combo-item"
              onMouseDown={(e) => { e.preventDefault(); onChange([...values, getValue(o)]); setQ(""); }}>
              {getLabel(o)}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function fmtErr(d) {
  if (!d) return "Errore. Riprova.";
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map(e => e?.msg || JSON.stringify(e)).join(" ");
  return d.msg || JSON.stringify(d);
}

// =============== LAYOUT ===============
function Shell({ children }) {
  const { user, logout } = useAuth();
  return (
    <div className="min-h-screen bg-pitch text-ink">
      <nav className="navbar" data-testid="main-navbar">
        <Link to="/" className="brand" data-testid="nav-brand">
          <span className="brand-mark">FM</span> FantaModel
        </Link>
        <div className="nav-links">
          <Link to="/index" data-testid="nav-index">Indice</Link>
          <Link to="/bonus" data-testid="nav-bonus">Bonus</Link>
          <Link to="/compare" data-testid="nav-compare">Confronto</Link>
          {user && <span className="user-pill" data-testid="user-email">{user.email}</span>}
          {user && <button className="btn-ghost" onClick={logout} data-testid="logout-btn">Esci</button>}
        </div>
      </nav>
      <main className="container">{children}</main>
    </div>
  );
}

function Protected({ children }) {
  const { user } = useAuth();
  if (user === null) return <div className="loading">Caricamento…</div>;
  if (!user) return <Navigate to="/login" replace />;
  return <Shell>{children}</Shell>;
}

// =============== AUTH PAGES ===============
function LoginPage() {
  const [tab, setTab] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const { setUser } = useAuth();
  const nav = useNavigate();

  const submit = async () => {
    setErr(""); setLoading(true);
    try {
      const url = tab === "login" ? `${API}/auth/login` : `${API}/auth/register`;
      const body = tab === "login" ? { email, password } : { email, password, name };
      const r = await axios.post(url, body);
      setUser(r.data.user);
      nav("/");
    } catch (e) {
      setErr(fmtErr(e.response?.data?.detail) || e.message);
    } finally { setLoading(false); }
  };

  return (
    <div className="auth-wrap">
      <div className="auth-card" data-testid="auth-card">
        <div className="auth-logo">⚽ FantaModel</div>
        <p className="auth-subtitle">Probabilità goal & assist con ML — Serie A</p>
        <div className="tabs">
          <button className={tab === "login" ? "tab tab-on" : "tab"} onClick={() => setTab("login")} data-testid="tab-login">Accedi</button>
          <button className={tab === "register" ? "tab tab-on" : "tab"} onClick={() => setTab("register")} data-testid="tab-register">Registrati</button>
        </div>
        {tab === "register" && (
          <input className="input" placeholder="Nome (opzionale)" value={name} onChange={e => setName(e.target.value)} data-testid="input-name" />
        )}
        <input className="input" type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} data-testid="input-email" />
        <input className="input" type="password" placeholder="Password (min 6)" value={password} onChange={e => setPassword(e.target.value)} data-testid="input-password" />
        {err && <div className="err" data-testid="auth-error">{err}</div>}
        <button className="btn-primary" onClick={submit} disabled={loading} data-testid="auth-submit">
          {loading ? "Attendere…" : tab === "login" ? "🔓 Entra" : "✨ Crea account"}
        </button>
        <p className="auth-hint">Demo admin: admin@fantamodel.com / admin123</p>
      </div>
    </div>
  );
}

// =============== HOME ===============
function HomePage() {
  const { user } = useAuth();
  const cards = [
    { to: "/index", title: "Indice Schierabilità", emoji: "📊", desc: "Calcola l'indice di schierabilità per uno o più giocatori e top per ruolo.", testid: "card-index" },
    { to: "/bonus", title: "Bonus Predictor", emoji: "🔮", desc: "Stima goal, assist, xG/xA e probabilità di bonus.", testid: "card-bonus" },
    { to: "/compare", title: "Confronto Giocatori", emoji: "⚔️", desc: "Confronta statistiche e probabilità tra due giocatori.", testid: "card-compare" },
  ];
  return (
    <div className="home">
      <div className="home-hero">
        <h1>Ciao {user?.name || "Allenatore"} 👋</h1>
        <p>Scegli uno strumento per la prossima giornata di Serie A.</p>
      </div>
      <div className="card-grid">
        {cards.map(c => (
          <Link to={c.to} key={c.to} className="feat-card" data-testid={c.testid}>
            <div className="feat-emoji">{c.emoji}</div>
            <h3>{c.title}</h3>
            <p>{c.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}

// =============== HELPERS ===============
function useApiData(url, deps = []) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let alive = true;
    axios.get(url).then(r => { if (alive) setData(r.data); }).catch(e => {
      if (alive) toast.error(fmtErr(e.response?.data?.detail) || e.message);
    });
    return () => { alive = false; };
    // eslint-disable-next-line
  }, deps);
  return data;
}

function ProgressBar({ value, color = "#22c55e" }) {
  const v = Math.max(0, Math.min(1, value || 0));
  return (
    <div className="pb"><div className="pb-fill" style={{ width: `${v * 100}%`, background: color }} /></div>
  );
}

// =============== BONUS PREDICTOR ===============
function BonusPage() {
  const players = useApiData(`${API}/players`);
  const [player, setPlayer] = useState("");
  const [team, setTeam] = useState("");
  const [opponent, setOpponent] = useState("");
  const [ha, setHa] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    if (!player) return;
    axios.get(`${API}/player-info?player=${encodeURIComponent(player)}`).then(r => {
      setTeam(r.data.team || "");
      setOpponent(r.data.auto_opponent || "");
      if (r.data.auto_ha) setHa(r.data.auto_ha);
    }).catch(() => {});
  }, [player]);

  const predict = async () => {
    if (!player || !team || !opponent) { toast.error("Compila tutti i campi"); return; }
    setLoading(true); setResult(null);
    try {
      const r = await axios.post(`${API}/predict/bonus`, { player, team, opponent, h_a: ha || null });
      setResult(r.data);
    } catch (e) {
      toast.error(fmtErr(e.response?.data?.detail) || e.message);
    } finally { setLoading(false); }
  };

  return (
    <div className="page">
      <h2 className="page-title">🔮 Bonus Predictor</h2>
      <p className="page-sub">Probabilità che un giocatore segni o faccia assist nella prossima partita.</p>
      <div className="form-grid">
        <div>
          <label>Giocatore</label>
          <SearchableSelect options={players} value={player} onChange={setPlayer} placeholder="Cerca giocatore..." testId="bonus-player-select" />
        </div>
        <div>
          <label>Squadra</label>
          <input className="input" value={team} onChange={e => setTeam(e.target.value)} data-testid="bonus-team" />
        </div>
        <div>
          <label>Avversario</label>
          <input className="input" value={opponent} onChange={e => setOpponent(e.target.value)} data-testid="bonus-opponent" />
        </div>
        <div>
          <label>Casa / Trasferta</label>
          <select className="input" value={ha} onChange={e => setHa(e.target.value)} data-testid="bonus-ha">
            <option value="">Auto</option>
            <option value="h">🏠 Casa</option>
            <option value="a">✈️ Trasferta</option>
          </select>
        </div>
      </div>
      <button className="btn-primary" onClick={predict} disabled={loading} data-testid="bonus-predict-btn">
        {loading ? "Calcolo…" : "⚡ Prevedi Bonus"}
      </button>

      {result && (
        <div className="result" data-testid="bonus-result">
          <h3>{result.player} ({result.team} vs {result.opponent})</h3>
          <div className="metrics-row">
            <div className="metric"><span>⚽ Goal</span><strong>{((result.goal_proba || 0) * 100).toFixed(1)}%</strong>
              <ProgressBar value={result.goal_proba} color="#3b82f6" /></div>
            <div className="metric"><span>👟 Assist</span><strong>{((result.assist_proba || 0) * 100).toFixed(1)}%</strong>
              <ProgressBar value={result.assist_proba} color="#10b981" /></div>
            <div className="metric metric-hot"><span>💎 Bonus Totale</span><strong>{((result.bonus_proba || 0) * 100).toFixed(1)}%</strong>
              <ProgressBar value={result.bonus_proba} color="#f59e0b" /></div>
          </div>

          {result.stats && (
            <div className="stats-grid">
              <div className="stat-card">
                <h4>📊 Statistiche stagione</h4>
                <ul>
                  <li>Presenze: <b>{result.stats.appearances}</b></li>
                  <li>Goal: <b>{result.stats.goals}</b></li>
                  <li>Assist: <b>{result.stats.assists}</b></li>
                  <li>xG medio: <b>{result.stats.xg_mean?.toFixed(2)}</b></li>
                  <li>xG ultime 5: <b>{result.stats.xg_last5?.toFixed(2)}</b></li>
                  <li>xA medio: <b>{result.stats.xa_mean?.toFixed(2)}</b></li>
                  <li>Tiri/match: <b>{result.stats.shots_per_match?.toFixed(1)}</b></li>
                </ul>
              </div>
              {result.stats.history_xg?.length > 0 && (
                <div className="stat-card">
                  <h4>📉 Andamento xG / Goal (ultime 10)</h4>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={result.stats.history_xg} margin={{ top: 10, right: 20, left: 0, bottom: 6 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="date" hide />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                      <Legend wrapperStyle={{ paddingTop: 8 }} iconType="line" />
                      <Line type="monotone" dataKey="xG" name="xG atteso" stroke="#3b82f6" strokeWidth={2.5} dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="goal" name="Goal segnati" stroke="#10b981" strokeWidth={2.5} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              {result.stats.history_xa?.length > 0 && (
                <div className="stat-card">
                  <h4>📈 Andamento xA / Assist (ultime 10)</h4>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={result.stats.history_xa} margin={{ top: 10, right: 20, left: 0, bottom: 6 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis dataKey="date" hide />
                      <YAxis stroke="#94a3b8" />
                      <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                      <Legend wrapperStyle={{ paddingTop: 8 }} iconType="line" />
                      <Line type="monotone" dataKey="xA" name="xA atteso" stroke="#10b981" strokeWidth={2.5} dot={{ r: 3 }} />
                      <Line type="monotone" dataKey="assist" name="Assist forniti" stroke="#f59e0b" strokeWidth={2.5} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// =============== COMPARE ===============
function ComparePage() {
  const players = useApiData(`${API}/players`);
  const [p1, setP1] = useState({ player: "", team: "", opponent: "", ha: "" });
  const [p2, setP2] = useState({ player: "", team: "", opponent: "", ha: "" });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const fillAuto = async (which, player) => {
    if (!player) return;
    const { data } = await axios.get(`${API}/player-info?player=${encodeURIComponent(player)}`);
    const update = { player, team: data.team || "", opponent: data.auto_opponent || "", ha: data.auto_ha || "" };
    if (which === 1) setP1(update); else setP2(update);
  };

  const compare = async () => {
    if (!p1.player || !p2.player) { toast.error("Seleziona entrambi i giocatori"); return; }
    setLoading(true); setResult(null);
    try {
      const r = await axios.post(`${API}/predict/compare`, {
        player1: p1.player, team1: p1.team, opponent1: p1.opponent, h_a1: p1.ha || null,
        player2: p2.player, team2: p2.team, opponent2: p2.opponent, h_a2: p2.ha || null,
      });
      setResult(r.data);
    } catch (e) {
      toast.error(fmtErr(e.response?.data?.detail) || e.message);
    } finally { setLoading(false); }
  };

  const PCard = ({ p, set, idx }) => (
    <div className="player-form">
      <h4>👤 Giocatore {idx}</h4>
      <SearchableSelect options={players} value={p.player} onChange={(v) => fillAuto(idx, v)} placeholder="Cerca giocatore..." testId={`compare-player${idx}`} />
      <input className="input" placeholder="Squadra" value={p.team} onChange={e => set({ ...p, team: e.target.value })} data-testid={`compare-team${idx}`} />
      <input className="input" placeholder="Avversario" value={p.opponent} onChange={e => set({ ...p, opponent: e.target.value })} data-testid={`compare-opp${idx}`} />
      <select className="input" value={p.ha} onChange={e => set({ ...p, ha: e.target.value })} data-testid={`compare-ha${idx}`}>
        <option value="">Auto</option>
        <option value="h">🏠 Casa</option>
        <option value="a">✈️ Trasferta</option>
      </select>
    </div>
  );

  const radarData = result ? (() => {
    const stats = [
      { key: "xg_mean", label: "xG medio", path: "stats.xg_mean" },
      { key: "xg_last5", label: "xG last5", path: "stats.xg_last5" },
      { key: "xa_last5", label: "xA last5", path: "stats.xa_last5" },
      { key: "goal_proba", label: "P.Goal", path: "goal_proba" },
      { key: "assist_proba", label: "P.Assist", path: "assist_proba" },
      { key: "bonus_proba", label: "P.Bonus", path: "bonus_proba" },
    ];
    const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? 0 : o[k]) ?? 0, obj) || 0;
    return stats.map(s => {
      const a = get(result.p1, s.path);
      const b = get(result.p2, s.path);
      const m = Math.max(Math.abs(a), Math.abs(b), 0.0001);
      return { stat: s.label, A: +(a / m).toFixed(3), B: +(b / m).toFixed(3), rawA: a, rawB: b };
    });
  })() : [];

  const radarTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const item = payload[0]?.payload;
    return (
      <div style={{ background: "#0f172a", border: "1px solid #334155", padding: "8px 12px", borderRadius: 8 }}>
        <div style={{ color: "#cbd5e1", fontSize: 12, marginBottom: 4 }}>{item.stat}</div>
        <div style={{ color: "#3b82f6" }}>{result.p1.player}: <b>{Number(item.rawA).toFixed(3)}</b></div>
        <div style={{ color: "#f59e0b" }}>{result.p2.player}: <b>{Number(item.rawB).toFixed(3)}</b></div>
      </div>
    );
  };

  return (
    <div className="page">
      <h2 className="page-title">⚔️ Confronto Giocatori</h2>
      <div className="compare-grid">
        <PCard p={p1} set={setP1} idx={1} />
        <PCard p={p2} set={setP2} idx={2} />
      </div>
      <button className="btn-primary" onClick={compare} disabled={loading} data-testid="compare-btn">
        {loading ? "Calcolo…" : "🔍 Confronta"}
      </button>

      {result && (
        <div className="result" data-testid="compare-result">
          <div className="winners">
            <span>⚽ Più probabilità Goal: <b>{(result.p1.goal_proba >= result.p2.goal_proba ? result.p1 : result.p2).player}</b></span>
            <span>👟 Più probabilità Assist: <b>{(result.p1.assist_proba >= result.p2.assist_proba ? result.p1 : result.p2).player}</b></span>
            <span>💎 Più probabilità Bonus: <b>{(result.p1.bonus_proba >= result.p2.bonus_proba ? result.p1 : result.p2).player}</b></span>
          </div>
          <div className="compare-cards">
            {[result.p1, result.p2].map((p, i) => (
              <div key={i} className="compare-card">
                <h3>{p.player}</h3>
                <div className="metric"><span>⚽ Goal</span><strong>{((p.goal_proba || 0) * 100).toFixed(1)}%</strong></div>
                <div className="metric"><span>👟 Assist</span><strong>{((p.assist_proba || 0) * 100).toFixed(1)}%</strong></div>
                <div className="metric"><span>💎 Bonus</span><strong>{((p.bonus_proba || 0) * 100).toFixed(1)}%</strong></div>
                <ul>
                  <li>xG: {p.stats?.xg_mean?.toFixed(2)} | last5 {p.stats?.xg_last5?.toFixed(2)}</li>
                  <li>xA: {p.stats?.xa_mean?.toFixed(2)} | last5 {p.stats?.xa_last5?.toFixed(2)}</li>
                  <li>Goal: {p.stats?.goals} | Assist: {p.stats?.assists} | Pres: {p.stats?.appearances}</li>
                </ul>
              </div>
            ))}
          </div>
          <div className="radar-wrap">
            <ResponsiveContainer width="100%" height={420}>
              <RadarChart data={radarData} outerRadius="78%">
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="stat" stroke="#cbd5e1" tick={{ fontSize: 13, fontWeight: 600 }} />
                <PolarRadiusAxis stroke="#475569" domain={[0, 1]} tick={false} axisLine={false} />
                <Radar name={result.p1.player} dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.45} strokeWidth={2} />
                <Radar name={result.p2.player} dataKey="B" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.45} strokeWidth={2} />
                <Tooltip content={radarTooltip} />
                <Legend wrapperStyle={{ paddingTop: 12 }} />
              </RadarChart>
            </ResponsiveContainer>
            <p className="radar-hint">📐 Valori normalizzati sul massimo tra i due giocatori per ogni metrica (per evidenziare le differenze relative).</p>
          </div>
        </div>
      )}
    </div>
  );
}

// =============== INDEX SCHIERABILITÀ ===============
function IndexPage() {
  const players = useApiData(`${API}/players`);
  const [selected, setSelected] = useState([]);
  const [loadingI, setLoadingI] = useState(false);
  const [loadingT, setLoadingT] = useState(false);
  const [indexResult, setIndexResult] = useState(null);
  const [topResult, setTopResult] = useState(null);

  const calc = async () => {
    if (!selected.length) { toast.error("Seleziona almeno un giocatore"); return; }
    setLoadingI(true); setIndexResult(null);
    try {
      const r = await axios.post(`${API}/predict/index`, { players: selected });
      setIndexResult(r.data);
    } catch (e) {
      toast.error(fmtErr(e.response?.data?.detail) || e.message);
    } finally { setLoadingI(false); }
  };

  const top = async () => {
    setLoadingT(true); setTopResult(null);
    try {
      const r = await axios.get(`${API}/predict/top-by-role`, { timeout: 360000 });
      setTopResult(r.data);
    } catch (e) {
      toast.error(fmtErr(e.response?.data?.detail) || e.message);
    } finally { setLoadingT(false); }
  };

  const renderTable = (data) => {
    if (!data?.rows?.length) return <div className="empty">Nessun risultato</div>;
    const cols = data.columns;
    let maxIdx = -Infinity, minIdx = Infinity;
    const idxCol = cols.includes("Index") ? "Index" : (cols.includes("fantavoto_pred") ? "fantavoto_pred" : null);
    if (idxCol) data.rows.forEach(r => { const v = parseFloat(r[idxCol]); if (!isNaN(v)) { if (v > maxIdx) maxIdx = v; if (v < minIdx) minIdx = v; } });
    return (
      <div className="tbl-wrap">
        <table className="tbl" data-testid="index-table">
          <thead><tr>{cols.map(c => <th key={c}>{c}</th>)}</tr></thead>
          <tbody>
            {data.rows.map((r, i) => (
              <tr key={i}>
                {cols.map(c => {
                  let v = r[c];
                  if (typeof v === "number") v = v.toFixed(1);
                  let cls = "";
                  if (idxCol === c) {
                    const f = parseFloat(r[c]);
                    if (f === maxIdx) cls = "cell-max";
                    else if (f === minIdx) cls = "cell-min";
                  }
                  return <td key={c} className={cls}>{v}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const ROLES = { P: "🧤 Portieri", D: "🛡 Difensori", C: "👟 Centrocampisti", A: "⚽ Attaccanti" };

  return (
    <div className="page">
      <h2 className="page-title">📈 Indice di Schierabilità</h2>
      <p className="page-sub">Calcola l'indice per i tuoi giocatori o vedi i top per ruolo per la prossima giornata.</p>

      <label>Seleziona giocatori</label>
      <SearchableMultiSelect options={players} values={selected} onChange={setSelected} placeholder="Cerca e aggiungi giocatori..." testId="index-multi" />
      <small className="hint">Digita per filtrare. Selezionati: {selected.length}</small>

      <div className="actions-row">
        <button className="btn-primary" onClick={calc} disabled={loadingI} data-testid="calc-index-btn">
          {loadingI ? "Calcolo…" : "🔮 Calcola Indice"}
        </button>
        <button className="btn-secondary" onClick={top} disabled={loadingT} data-testid="top-by-role-btn">
          {loadingT ? "Calcolo… (può richiedere 1-2 min)" : "🏆 Top per Ruolo"}
        </button>
      </div>
      <small className="hint">⚠️ Top per Ruolo è un calcolo intensivo su tutti i giocatori — può richiedere 1-2 minuti.</small>

      {indexResult && (
        <div className="result">
          <h3>🔮 Risultati Predizione</h3>
          {renderTable(indexResult)}
        </div>
      )}

      {topResult && (
        <div className="result">
          <h3>🏆 Top per Ruolo</h3>
          {Object.keys(ROLES).map(role => topResult[role] && (
            <div key={role} className="role-section">
              <h4>{ROLES[role]}</h4>
              {renderTable(topResult[role])}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// =============== APP ===============
export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Toaster richColors position="top-right" />
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<Protected><HomePage /></Protected>} />
          <Route path="/index" element={<Protected><IndexPage /></Protected>} />
          <Route path="/bonus" element={<Protected><BonusPage /></Protected>} />
          <Route path="/compare" element={<Protected><ComparePage /></Protected>} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
