3D Single-Particle Tracking Tool (V2 Lite)

Enthalten:
- Automatisches Tracking (Auto‑Mode): parametrisiert `search_range` und `memory` per Scan
- Visualisierung: Raw, Time‑resolved (plasma, echte Zeit), SNR (cividis) – jeweils SVG
- Export: Excel (Summary + pro Track Sheet)

Voraussetzungen:
- Python 3.9+
- Pakete: pandas, numpy, trackpy, matplotlib, openpyxl

Start (GUI):
- `python V2/tracking_tool.py`

Ablauf:
1. CSV (ThunderSTORM) auswählen und laden (Pre‑Filter empfohlen)
2. Automodus aktiv lassen (empfohlen) oder Parameter manuell setzen
3. Tracking starten
4. Alles exportieren (SVG + Excel)

Ausgabeordnerstruktur:
- `3D_Tracking_Results/01_Raw_Tracks/`
- `3D_Tracking_Results/02_Time_Resolved_Tracks/`
- `3D_Tracking_Results/03_SNR_Tracks/`
- `3D_Tracking_Results/04_Tracks/` (Excel)

