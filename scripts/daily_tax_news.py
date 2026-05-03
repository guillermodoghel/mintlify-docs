#!/usr/bin/env python3
"""
Daily Argentine tax news aggregator for facture.ar/novedades.

Two passes per run:
  1. Multi-source: generates an article if 2+ papers cover the same tax topic.
  2. High-relevance singles: generates articles for individually important items
     (vencimientos, nuevas obligaciones, nuevos formularios, etc.) even if
     only one source covers them.

Maximum 3 new articles per run to avoid flooding the novedades section.
"""

import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import anthropic
import feedparser

REPO_ROOT = Path(__file__).parent.parent
NOVEDADES_DIR = REPO_ROOT / "es" / "novedades"
DOCS_JSON = REPO_ROOT / "docs.json"

ART = timezone(timedelta(hours=-3))  # Argentina (no DST)
MAX_ARTICLES_PER_RUN = 3

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
    t = text.lower()
    return any(kw in t for kw in TAX_KEYWORDS)


def slug_from(text: str) -> str:
    """Convert text to a safe kebab-case slug."""
    s = text.lower()
    s = re.sub(r"[áàä]", "a", s)
    s = re.sub(r"[éèë]", "e", s)
    s = re.sub(r"[íìï]", "i", s)
    s = re.sub(r"[óòö]", "o", s)
    s = re.sub(r"[úùü]", "u", s)
    s = re.sub(r"ñ", "n", s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")[:60]


def fetch_news() -> list[dict]:
    """Fetch last 24h tax-related articles from all RSS feeds."""
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


def already_exists(slug_base: str) -> bool:
    """Return True if an article with this slug base was already written today."""
    today = date.today().isoformat()
    # Exact match with today's date suffix
    if (NOVEDADES_DIR / f"{slug_base}-{today}.mdx").exists():
        return True
    # Loose match: any existing file whose name starts with slug_base
    return any(NOVEDADES_DIR.glob(f"{slug_base}*.mdx"))


def analyze_common_themes(articles: list[dict]) -> dict | None:
    """Pass 1 — find a topic covered by 2+ sources."""
    if len(articles) < 2:
        return None

    client = anthropic.Anthropic()
    articles_text = "\n\n".join(
        f"[{a['source']}] {a['title']}\n{a['summary']}" for a in articles
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{
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
        }],
    )

    try:
        return json.loads(response.content[0].text.strip())
    except Exception as e:
        print(f"Error parsing multi-source analysis: {e}", file=sys.stderr)
        return None


def find_high_relevance_singles(articles: list[dict], exclude_slugs: list[str]) -> list[dict]:
    """Pass 2 — find individually important articles not already covered.

    Returns a list of dicts: {title, source, summary, slug, reason}
    """
    if not articles:
        return []

    client = anthropic.Anthropic()

    articles_text = "\n\n".join(
        f"[{i}] [{a['source']}] {a['title']}\n{a['summary']}"
        for i, a in enumerate(articles)
    )

    already_covered = ", ".join(exclude_slugs) if exclude_slugs else "ninguno"

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Sos el editor del blog impositivo de facture.ar, una plataforma de facturación electrónica para empresas y contadores argentinos.

Analizá estas noticias y seleccioná hasta 2 artículos que sean MUY relevantes para publicar en el blog, aunque una sola fuente los cubra. Priorizá:
- Vencimientos o fechas límite próximas (monotributo, ganancias, etc.)
- Nuevos formularios o procedimientos de ARCA/AFIP
- Cambios en alícuotas o categorías
- Nuevas obligaciones para contribuyentes
- Información práctica y accionable para el mes en curso

NO selecciones artículos sobre: economía general, inflación, mercados, agro, política, noticias sin impacto directo en obligaciones fiscales.

Temas ya cubiertos hoy (NO repetir): {already_covered}

NOTICIAS:
{articles_text}

Respondé SOLO con JSON válido:
{{
  "articulos": [
    {{
      "indice": 0,
      "titulo_articulo": "título sugerido para el artículo del blog",
      "slug": "nombre-kebab-case-sin-fecha",
      "resumen": "qué cubre y por qué es relevante para los usuarios de facture.ar"
    }}
  ]
}}

Si ninguno es suficientemente relevante, respondé: {{"articulos": []}}""",
        }],
    )

    try:
        data = json.loads(response.content[0].text.strip())
        result = []
        for item in data.get("articulos", []):
            idx = item.get("indice")
            if idx is not None and 0 <= idx < len(articles):
                source_article = articles[idx]
                result.append({
                    "title": item["titulo_articulo"],
                    "slug": item["slug"],
                    "summary": item["resumen"],
                    "source": source_article["source"],
                    "source_articles": [source_article],
                })
        return result
    except Exception as e:
        print(f"Error parsing single-relevance analysis: {e}", file=sys.stderr)
        return []


def generate_article(topic: str, sources: list[str], summary: str, source_articles: list[dict]) -> str:
    """Generate original MDX article."""
    client = anthropic.Anthropic()

    sources_detail = "\n".join(
        f"- {a['source']}: {a['title']} — {a['summary']}" for a in source_articles
    )
    today = date.today().strftime("%d/%m/%Y")

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Escribí un artículo original en español sobre esta novedad impositiva argentina para el blog de facture.ar.

TEMA: {topic}
RESUMEN: {summary}
FUENTE(S): {", ".join(sources)}
DETALLE DE LAS NOTICIAS FUENTE:
{sources_detail}
FECHA: {today}

FORMATO REQUERIDO — MDX con frontmatter YAML:
- Empezá con frontmatter (title y description)
- Tono profesional y directo, dirigido a contadores y empresas argentinas
- Usá vos/ustedes (no tuteo ni usted)
- Secciones con ## y ### según corresponda
- Usá <Info> para aclaraciones importantes, <Warning> para alertas o fechas límite
- Usá tablas o <Steps> si hay fechas de vencimiento, montos o procedimientos paso a paso
- Citá normas concretas (RG, decretos, resoluciones) si las hay
- Terminá con "## ¿Qué cambia en Facturear?" — indicá si hay impacto en la plataforma (si no hay, decilo claro)
- Extensión: 500-900 palabras de contenido
- NO inventés datos — basate solo en la información provista
- El artículo debe ser original y útil, no un resumen de las notas fuente

Respondé SOLO el MDX, comenzando con ---""",
        }],
    )

    content = response.content[0].text.strip()
    if not content.startswith("---"):
        content = (
            f"---\ntitle: '{topic}'\ndescription: 'Novedad impositiva del {date.today().isoformat()}'\n---\n\n"
            + content
        )
    return content


def save_article(slug: str, mdx_content: str) -> None:
    """Write MDX file and update docs.json navigation."""
    output_path = NOVEDADES_DIR / f"{slug}.mdx"
    output_path.write_text(mdx_content, encoding="utf-8")
    print(f"  Written: {output_path.name}")

    page_path = f"es/novedades/{slug}"
    with open(DOCS_JSON, encoding="utf-8") as f:
        docs = json.load(f)

    for tab in docs["navigation"]["tabs"]:
        for group in tab.get("groups", []):
            if group.get("group") == "Novedades Impositivas":
                if page_path not in group["pages"]:
                    group["pages"].insert(0, page_path)
                with open(DOCS_JSON, "w", encoding="utf-8") as f:
                    json.dump(docs, f, indent=2, ensure_ascii=False)
                    f.write("\n")
                print(f"  docs.json updated: {page_path}")
                return

    print("Warning: 'Novedades Impositivas' group not found in docs.json", file=sys.stderr)


def main() -> None:
    print(f"[{datetime.now(tz=ART).isoformat()}] Starting daily tax news aggregator")

    print("\nFetching RSS feeds...")
    articles = fetch_news()
    print(f"Found {len(articles)} tax-related articles in the last 24h:")
    for a in articles:
        print(f"  [{a['source']}] {a['title']}")

    if not articles:
        print("Nothing to process today.")
        return

    today = date.today().isoformat()
    articles_written = 0
    covered_slugs: list[str] = []

    # ── Pass 1: multi-source common theme ────────────────────────────────────
    print("\n[Pass 1] Checking for common themes across 2+ sources...")
    analysis = analyze_common_themes(articles)

    if analysis and analysis.get("hay_tema_comun"):
        topic = analysis["tema_principal"]
        sources = analysis["fuentes"]
        summary = analysis["resumen_del_hecho"]
        slug_base = slug_from(analysis.get("slug", "novedad-impositiva"))
        slug = f"{slug_base}-{today}"

        print(f"  Common topic: {topic} (sources: {', '.join(sources)})")

        if already_exists(slug_base):
            print(f"  Already exists — skipping.")
        else:
            mdx = generate_article(topic, sources, summary, articles)
            save_article(slug, mdx)
            covered_slugs.append(slug_base)
            articles_written += 1
    else:
        print("  No common topic found across 2+ sources.")

    # ── Pass 2: high-relevance singles ───────────────────────────────────────
    remaining_slots = MAX_ARTICLES_PER_RUN - articles_written
    if remaining_slots <= 0:
        print("\n[Pass 2] Skipped — daily limit reached.")
    else:
        print(f"\n[Pass 2] Looking for up to {remaining_slots} high-relevance single-source articles...")
        singles = find_high_relevance_singles(articles, covered_slugs)

        if not singles:
            print("  No individually relevant articles found.")
        else:
            for item in singles[:remaining_slots]:
                slug_base = slug_from(item["slug"])
                slug = f"{slug_base}-{today}"

                if already_exists(slug_base):
                    print(f"  Already exists ({slug_base}) — skipping.")
                    continue

                print(f"  Writing: {item['title']} [{item['source']}]")
                mdx = generate_article(
                    topic=item["title"],
                    sources=[item["source"]],
                    summary=item["summary"],
                    source_articles=item["source_articles"],
                )
                save_article(slug, mdx)
                covered_slugs.append(slug_base)
                articles_written += 1

    print(f"\nDone. {articles_written} article(s) written today.")


if __name__ == "__main__":
    main()
