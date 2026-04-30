# FantaModel — PRD

## Original Problem
User has a Streamlit app (3 pages) for Serie A goal/assist probabilities & schierabilità index. Uses ML models from GitHub repo https://github.com/malo93777/fantamodel. Wants to migrate to a real web app with serious login (JWT + Google) + €1/month subscription. Free registration with optional paid plan. Repo code MUST NOT be modified.

## Stack
- Backend: FastAPI + MongoDB (motor) + Python ML (joblib, catboost, xgboost, sklearn, statsmodels)
- Frontend: React 19 + recharts + sonner + react-router
- Auth: JWT (bcrypt + httpOnly cookies)
- ML: copied verbatim from upstream repo into /app/backend/fantamodel (src/, models/, dataset/, scaler/)

## What's Implemented (Phase 1 — Apr 2026)
- JWT email/password auth (register, login, logout, /me) with admin seeding
- Bonus Predictor page (goal_proba, assist_proba, bonus_proba, history charts xG/xA last 10)
- Compare Players page (radar + side-by-side stats + winners)
- Indice Schierabilità page (multi-select + table with Index/fantavoto highlighted) + Top by Role (P/D/C/A)
- Editorial dark "pitch" theme with Bebas Neue + Manrope fonts, gold/green accents

## P0 Backlog (Phase 2 — next session)
- Google OAuth via Emergent-managed Google Auth
- Stripe €1/month subscription + paywall on premium features
- User profile / settings page

## P1 Backlog
- Cache ML predictions per (player, team, opponent, h_a) for X minutes
- Mobile UI polish for tables (currently scroll horizontally)
- Export results as CSV/PDF

## Personas
- **Fantacalcio coach**: needs goal/assist forecasts + schierabilità per giornata
- **Power user**: compares players to decide swaps
