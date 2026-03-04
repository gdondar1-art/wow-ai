import os, json, hashlib, time
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from urllib.parse import quote_plus

def load_env(path="/opt/wow-ai/.env"):
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k,v=line.split("=",1)
            os.environ.setdefault(k.strip(), v.strip())

load_env()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOKEN = os.getenv("WOW_TOKEN", "")
BOOKING_AID = os.getenv("BOOKING_AID", "")
GYG_PARTNER = os.getenv("GYG_PARTNER", "")

SEED_PATH = "/opt/wow-ai/seed/destinations.json"
LOG_PATH = "/opt/wow-ai/logs/ai.log"

CACHE: Dict[str, Any] = {}
CACHE_TTL_SEC = 1800

app = FastAPI()

class PlanRequest(BaseModel):
    message: str
    context: Optional[str] = "general"
    lang: Optional[str] = "de"

def log(msg: str):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except:
        pass

def load_seed() -> List[Dict[str, Any]]:
    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

SEED = load_seed()

def cache_key(req: PlanRequest) -> str:
    raw = f"{req.lang}|{req.context}|{req.message}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def booking_link(q: str) -> str:
    return f"https://www.booking.com/searchresults.html?ss={quote_plus(q)}&aid={BOOKING_AID}"

def gyg_link(q: str) -> str:
    return f"https://www.getyourguide.com/s/?q={quote_plus(q)}&partner_id={GYG_PARTNER}"

def surprise_message() -> str:
    if not SEED:
        return "Überrasch mich: 3 Tage Wellness in Deutschland, ruhig, Budget 500€."
    pick = SEED[int(time.time()) % len(SEED)]["name"]
    return f"Überrasch mich: 3 Tage Kurztrip nach {pick} in Deutschland, ruhig, Wellness/Erholung."

INTENT_PROMPT_DE = """Du bist ein präziser Intent-Parser für Reisen in Deutschland.
Extrahiere aus der Nutzernachricht strukturierte Daten und antworte ausschließlich als gültiges JSON:

{
  "niche":"health|wellness|eco|family|hidden|mini|other",
  "budget_eur": number|null,
  "days": number|null,
  "region": string|null,
  "start_city": string|null,
  "group_type":"solo|couple|family|multi_gen|friends|unknown",
  "must_haves": string[],
  "avoid": string[],
  "notes": string
}

Regeln:
- Wenn Budget/Tage fehlen: null.
- Gesundheitsreisen (Kur, Therme, Stressabbau) => "health".
- Wellness (Spa, Yoga, Detox) => "wellness".
- Nur JSON, keine Erklärungen.
"""

PLAN_PROMPT_DE = """Du bist WOW KI Reisen. Erstelle 3 Vorschläge in Deutschland basierend auf Intent.
Antworte ausschließlich als gültiges JSON mit folgendem Schema:

{
  "intent": { },
  "recommendations": [
    {
      "title": "",
      "best_for": "",
      "why": "",
      "itinerary": [
        {"day": 1, "plan": ["", "", ""]},
        {"day": 2, "plan": ["", "", ""]},
        {"day": 3, "plan": ["", "", ""]}
      ],
      "estimated_cost": {
        "total_eur": 0,
        "breakdown": [
          {"item":"Unterkunft","eur":0},
          {"item":"Aktivitäten","eur":0},
          {"item":"Essen","eur":0},
          {"item":"Transport lokal","eur":0}
        ]
      },
      "hotels": [
        {"name":"","city":"","price_hint":"€|€€|€€€","booking_query":""},
        {"name":"","city":"","price_hint":"€|€€|€€€","booking_query":""}
      ],
      "activities": [
        {"name":"","city":"","time_hint":"","tour_query":""},
        {"name":"","city":"","time_hint":"","tour_query":""}
      ]
    }
  ],
  "ui": { "disclaimer": "" }
}

Kurallar:
- recommendations: genau 3 Einträge.
- Nur JSON, kein Markdown.
- booking_query / tour_query sind Suchbegriffe, keine URLs.
"""

def call_openai_json(system_text: str, user_text: str) -> Any:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_text},
            {"role":"user","content":user_text}
        ]
    )
    return json.loads(r.choices[0].message.content.strip())

def safe_fallback(req: PlanRequest) -> Dict[str, Any]:
    picks = SEED[:3] if len(SEED) >= 3 else [{"name":"Baden-Baden"},{"name":"Schwarzwald"},{"name":"Allgäu"}]
    recs = []
    for p in picks:
        city = p["name"]
        recs.append({
            "title": f"{city}: Kurztrip mit Erholung",
            "best_for": "Ruhig, einfach, ohne Stress",
            "why": "Schnelle Inspiration (Fallback).",
            "itinerary": [
                {"day": 1, "plan": ["Anreise & Check-in", "Spaziergang", "Ruhiges Abendessen"]},
                {"day": 2, "plan": ["Wellness/Therme", "Naturroute leicht", "Café/Altstadt"]},
                {"day": 3, "plan": ["Frühstück", "Kurpark/See", "Abreise"]},
            ],
            "estimated_cost": {"total_eur": 0, "breakdown": [
                {"item":"Unterkunft","eur":0},{"item":"Aktivitäten","eur":0},{"item":"Essen","eur":0},{"item":"Transport lokal","eur":0}
            ]},
            "hotels": [
                {"name":"Zentrales Hotel","city":city,"price_hint":"€€","booking_query":f"Hotel {city}"},
                {"name":"Wellnesshotel","city":city,"price_hint":"€€€","booking_query":f"Wellnesshotel {city} Therme"}
            ],
            "activities": [
                {"name":"Therme/Spa","city":city,"time_hint":"2–4 Std.","tour_query":f"{city} Therme Eintritt"},
                {"name":"Geführter Spaziergang","city":city,"time_hint":"1–2 Std.","tour_query":f"{city} geführter Spaziergang"}
            ]
        })
    return {"intent": {"language": req.lang, "context": req.context}, "recommendations": recs, "ui": {"disclaimer":"KI-Fallback. Details bitte prüfen."}}

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/plan")
def plan(req: PlanRequest, x_wow_token: str = Header(None)):
    if TOKEN and x_wow_token != TOKEN:
        raise HTTPException(401, "invalid token")

    msg = (req.message or "").strip()
    if msg == "" or msg.lower() in ["überrasch mich","uberrasch mich","surprise","überrasche mich"]:
        msg = surprise_message()
    req.message = msg

    ck = cache_key(req)
    now = time.time()
    if ck in CACHE and (now - CACHE[ck]["ts"] < CACHE_TTL_SEC):
        return CACHE[ck]["data"]

    t0 = time.time()
    try:
        intent = call_openai_json(INTENT_PROMPT_DE, req.message)
        if not intent.get("days"):
            intent["days"] = 3
        payload = json.dumps({"intent": intent, "context": req.context}, ensure_ascii=False)
        data = call_openai_json(PLAN_PROMPT_DE, payload)
    except Exception as e:
        log(f"[plan_error] {e}")
        data = safe_fallback(req)

    # affiliate injection
    for rec in data.get("recommendations", []):
        for h in rec.get("hotels", []):
            q = (h.get("booking_query") or "").strip() or f"Hotel {rec.get('title','')}"
            h["booking_link"] = booking_link(q)
        for a in rec.get("activities", []):
            q = (a.get("tour_query") or "").strip() or rec.get("title","")
            a["activity_link"] = gyg_link(q)

    data.setdefault("ui", {})
    data["ui"].setdefault("disclaimer", "Hinweis: KI-Vorschläge zur Inspiration. Links können Affiliate-Links sein.")

    dt = int((time.time() - t0) * 1000)
    log(f"[ok] {dt}ms context={req.context}")
    CACHE[ck] = {"ts": now, "data": data}
    return data
