# Audit Narrative (Simulated)

Scope
- Retail Forecast AIMS demo covering datasets, models, metrics, and operational safeguards.

Cadence
- Daily evidence cycle (simulated) with KPIs review (MAPE/RMSE), dataset changes, incidents, and CAPA status.
- Monthly internal audit (simulated) with independence attested.
- Quarterly management review (simulated) with improvement actions recorded.

How to Review
1. Launch the AIMS dashboard (`forecast_dashboard.py`) with `AIMS_DEMO=1 AIMS_DASHBOARD=1`.
2. Inspect KPIs and run `Run Simulated Cycle` to append evidence to `docs/evidence/`.
3. Click `Record Daily Log Hash` to append a SHA256 hash to `logs/daily_hashes.txt`.
4. For read-only QC, use helpers in `src/aims/qc.py` for bias drift and baseline comparison.
5. Optionally run `scripts/encrypt_evidence_demo.py` with `AIMS_ENCRYPT_DEMO=1` to encrypt evidence for demo purposes.

Caveats
- All artifacts are labeled "Simulated" and intended for portfolio demonstrations only.
- No blocking changes are introduced; flags are off by default; flows remain unchanged without flags.
