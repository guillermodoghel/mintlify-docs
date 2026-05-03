#!/usr/bin/env python3
"""
Daily Argentine tax news aggregator for facture.ar/novedades.

Fetches RSS feeds from major Argentine newspapers, identifies tax topics
covered by 2+ sources, and generates an original MDX article.
"""

import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import anthropic
import feedparser

REPO_ROOT = Path(__file__).parent.parent
NOVEDADES_DIR = REPO_ROOT / "es" / "novedades"
DOCS_JSON = REPO_ROOT / "docs.json"

# Argentina is UTC-3 (no DST)
ART = timezone(timedelta(hours=-3))

RSS_FEEDS = {
    "La Nación": [
        "https://www.lanacion.com.ar/arc/outboundfeeds/rss/category/economia/",
        "https://www.lanacion.com.ar/rss/economia.xml",
    ],
    "Infobae": [
        "https://www.infobae.com/arc/outboundfeeds/rss/category/economia/",
        "https://www.infobae.com/feeds/rss/economia/",
    ],
    "Ámbito Financiero": [
        "https://www.ambito.com/rss/economia",
        "https://www.ambito.com/rss/pages/economia.xml",
    ],
    "El Cronista": [
        "https://www.cronista.com/rss/economia.xml",
        "https://www.cronista.com/feed/",
    ],
    "iProfesional": [
        "https://www.iprofesional.com/rss/impuestos",
        "https://www.iprofesional.com/feeds/impuestos.xml",
    ],
    "Clarín": [
        "https://www.clarin.com/rss/economia/",
    ],
}

# Keywords that must appear to consider an article tax-related.
# Kept specific to avoid false positives from agro/finance articles.
TAX_KEYWORDS = [
    "afip", "arca ", "arca:", "arca-",
    "impuesto a las ganancias", "impuesto al valor agregado",
    "impositivo", "tributario", "régimen tributario",
    " iva ", "iva:", "iva,",
    "bienes personales", "monotributo", "monotributista",
    "blanqueo", "moratoria impositiva", "moratoria fiscal",
    "vencimiento impositivo", "vencimiento fiscal", "vence el plazo",
    "declaración jurada", "resolución general afip", "resolución general arca",
    " rg afip", " rg arca", "rg n°", "rg nro",
    "ingresos brutos", "agip", "arba",
    "crédito fiscal", "débito fiscal", "deducción impositiva",
    "mínimo no imponible", "plan de pago afip", "plan de pago arca",
    "régimen simplificado", "responsable inscripto",
    "factura electrónica", "comprobante electrónico",
    "clave fiscal", "cuit", "cuil",
    "retención impositiva", "retención de ganancias", "retención de iva",
    "percepción de iva", "percepción de ingresos",
]


def is_tax_related(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in TAX_KEYWORDS)


def fetch_news() -> list[dict]:
    """Fetch last 24h tax news from all RSS feeds."""
    cutoff = datetime.now(tz=ART) - timedelta(hours=24)
    articles = []

    for source, urls in RSS_FEEDS.items():
        feed = None
        for url in urls:
            try:
                parsed = feedparser.parse(url)
                if parsed.entries:
                    feed = parsed
                    break
            except Exception as e:
                print(f"  {source} ({url}): {e}", file=sys.stderr)

        if not feed:
            print(f"  {source}: no feed available", file=sys.stderr)
            continue

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            summary = re.sub(r"<[^>]+>", "", getattr(entry, "summary", "") or "").strip()

            if not is_tax_related(title + " " + summary):
                continue

            # Date filter — skip articles older than 24h if we can determine the date
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if pub_date < cutoff:
                        continue
                except Exception:
                    pass

            articles.append({
                "source": source,
                "title": title,
                "summary": summary[:600],
                "url": getattr(entry, "link", ""),
                "pub_date": pub_date.isoformat() if pub_date else "unknown",
            })

    return articles


def analyze_themes(articles: list[dict]) -> dict | None:
    """Ask Claude if 2+ sources cover the same tax topic. Returns analysis dict or None."""
    if len(articles) < 2:
        return None

    client = anthropic.Anthropic()

    articles_text = "\n\n".join(
        f"[{a['source']}] {a['title']}\n{a['summary']}" for a in articles
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""Analizá estas noticias impositivas argentinas del día y determiná si hay 2 o más fuentes distintas que cubran el MISMO hecho o novedad impositiva concreta.

NOTICIAS:
{articles_text}

Respondé SOLO con JSON válido, sin texto adicional:
{{
  "hay_tema_comun": true,
  "tema_principal": "título breve del tema común",
  "fuentes": ["fuente1", "fuente2"],
  "resumen_del_hecho": "2-3 oraciones sobre qué pasó exactamente, con datos concretos",
  "slug": "nombre-kebab-case-descriptivo-sin-fecha"
}}

Si no hay tema común entre 2+ fuentes, respondé: {{"hay_tema_comun": false}}""",
            }
        ],
    )

    try:
        return json.loads(response.content[0].text.strip())
    except Exception as e:
        print(f"Error parsing analysis: {e}\nRaw: {response.content[0].text}", file=sys.stderr)
        return None


def generate_article(topic: str, sources: list[str], summary: str, articles: list[dict]) -> str:
    """Generate original MDX article about the topic."""
    client = anthropic.Anthropic()

    relevant = [a for a in articles if a["source"] in sources]
    sources_detail = "\n".join(f"- {a['source']}: {a['title']} — {a['summary']}" for a in relevant)
    today = date.today().strftime("%d/%m/%Y")

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": f"""Escribí un artículo original en español sobre esta novedad impositiva argentina para el blog de facture.ar.

TEMA: {topic}
RESUMEN DEL HECHO: {summary}
FUENTES QUE LO CUBREN: {", ".join(sources)}
DETALLE:
{sources_detail}
FECHA: {today}

FORMATO REQUERIDO — MDX con frontmatter YAML:
- Empezá con frontmatter (title y description)
- Tono profesional y directo, dirigido a contadores y empresas argentinas
- Usá vos/ustedes (no tuteo ni usted)
- Secciones con ## y ### según corresponda
- Usá <Info> para aclaraciones importantes, <Warning> para alertas
- Usá tablas o <Steps> si hay fechas de vencimiento o procedimientos
- Citá normas concretas (RG, decretos, resoluciones) si las hay en el texto
- Terminá con una sección "## ¿Qué cambia en Facturear?" indicando si hay impacto en la plataforma (si no hay, decilo claramente)
- Extensión: 500-900 palabras de contenido
- NO inventés información — basate solo en lo que te doy
- El artículo debe ser original, no un resumen de las notas fuente

Respondé SOLO el MDX, comenzando con ---""",
            }
        ],
    )

    content = response.content[0].text.strip()
    if not content.startswith("---"):
        today_iso = date.today().isoformat()
        content = f"---\ntitle: '{topic}'\ndescription: 'Novedad impositiva del {today_iso}'\n---\n\n" + content
    return content


def add_to_docs_json(slug: str) -> None:
    """Insert new article at the top of the Novedades Impositivas group."""
    with open(DOCS_JSON, encoding="utf-8") as f:
        docs = json.load(f)

    page_path = f"es/novedades/{slug}"

    for tab in docs["navigation"]["tabs"]:
        for group in tab.get("groups", []):
            if group.get("group") == "Novedades Impositivas":
                if page_path not in group["pages"]:
                    group["pages"].insert(0, page_path)
                with open(DOCS_JSON, "w", encoding="utf-8") as f:
                    json.dump(docs, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                print(f"docs.json updated: added {page_path}")
                return

    print("Warning: 'Novedades Impositivas' group not found in docs.json", file=sys.stderr)


def main() -> None:
    print(f"[{datetime.now(tz=ART).isoformat()}] Starting daily tax news aggregator")

    print("Fetching RSS feeds...")
    articles = fetch_news()
    print(f"Found {len(articles)} tax-related articles in the last 24h")

    if not articles:
        print("Nothing to process today.")
        return

    for a in articles:
        print(f"  [{a['source']}] {a['title']}")

    print("\nAnalyzing for common themes...")
    analysis = analyze_themes(articles)

    if not analysis or not analysis.get("hay_tema_comun"):
        print("No common topic across 2+ sources today — no article generated.")
        return

    topic = analysis["tema_principal"]
    sources = analysis["fuentes"]
    summary = analysis["resumen_del_hecho"]
    slug_base = re.sub(r"[^a-z0-9-]", "", analysis.get("slug", "novedad-impositiva").lower())
    slug = f"{slug_base}-{date.today().isoformat()}"

    print(f"\nCommon topic: {topic}")
    print(f"Covered by: {', '.join(sources)}")

    output_path = NOVEDADES_DIR / f"{slug}.mdx"
    if output_path.exists():
        print(f"Article already exists: {output_path.name} — skipping.")
        return

    print("\nGenerating article...")
    mdx_content = generate_article(topic, sources, summary, articles)

    output_path.write_text(mdx_content, encoding="utf-8")
    print(f"Article written: {output_path.name}")

    add_to_docs_json(slug)
    print("\nDone.")


if __name__ == "__main__":
    main()
