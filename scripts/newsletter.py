#!/usr/bin/env python3
"""
Newsletter Facturear: builds every index page from the notes' frontmatter.

The notes live in es/newsletter/notas/<slug>.mdx, one per news item, with this frontmatter:

  title, description   what Mintlify shows
  date                 YYYY-MM-DD, publication date (ART)
  category             one of CATEGORIES
  impact               alto | medio | ninguno: how much it changes invoicing with Facturear
  impact_note          one sentence on what changes for someone who invoices (alto/medio)
  sources              papers that covered it (optional)
  tag                  set here from `impact`, shows next to the title in the sidebar

`python scripts/newsletter.py` regenerates, from scratch and idempotently:
  - es/newsletter/index.mdx               the front page, laid out like a newspaper
  - es/newsletter/impacto-facturear.mdx   everything that changes how people invoice
  - es/newsletter/temas/<category>.mdx    one index per topic
  - es/newsletter/archivo/<yyyy-mm>.mdx   one index per month
  - the impact callout and "Más sobre ..." block inside each note
  - the Newsletter tab in docs.json, the newsletter block in llms.txt and newsletter-feed.json

Stdlib only: the daily GitHub Action imports it right after writing new notes.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
NEWSLETTER_DIR = REPO_ROOT / "es" / "newsletter"
NOTES_DIR = NEWSLETTER_DIR / "notas"
DOCS_JSON = REPO_ROOT / "docs.json"
LLMS_TXT = REPO_ROOT / "llms.txt"
# Read by facture.ar to build the Monday email; keep the shape stable
FEED_JSON = REPO_ROOT / "newsletter-feed.json"
FEED_SIZE = 80
SITE = "https://docs.facture.ar"
TAB_NAME = "Newsletter"

ART = timezone(timedelta(hours=-3))

# (key, label, one-line description, evergreen guide to point to)
CATEGORIES = [
    ("facturacion", "Facturación electrónica", "Comprobantes, CAE, nuevos obligados y cambios en cómo se factura.", "/es/fundamentos/que-es-factura-electronica"),
    ("vencimientos", "Vencimientos", "Fechas límite, prórrogas y calendarios de ARCA y las provincias.", None),
    ("monotributo", "Monotributo", "Escalas, cuotas, topes de facturación y recategorizaciones.", "/es/monotributo/categorias-tabla"),
    ("iva", "IVA", "Alícuotas, crédito fiscal, percepciones y reintegros.", "/es/iva/que-es-iva"),
    ("ganancias", "Ganancias", "Escalas, deducciones, retenciones y declaraciones juradas.", "/es/ganancias/que-es-ganancias"),
    ("bienes-personales", "Bienes Personales", "Mínimos, alícuotas y declaración anual.", "/es/bienes-personales/que-es"),
    ("iibb", "Ingresos Brutos y provincias", "ARBA, AGIP, Convenio Multilateral y tributos locales.", "/es/ingresos-brutos/que-es-iibb"),
    ("laboral", "Laboral y previsional", "Contribuciones, sueldos, aguinaldo y aportes.", "/es/laboral/cargas-sociales"),
    ("arca", "ARCA y procedimientos", "Planes de pago, fiscalizaciones, regímenes de información y trámites.", "/es/tramites/clave-fiscal"),
    ("economia", "Economía y contexto", "Recaudación, reformas y contexto económico con efecto fiscal.", None),
]
CATEGORY = {key: (label, desc, guide) for key, label, desc, guide in CATEGORIES}
IMPACTS = ("alto", "medio", "ninguno")
IMPACT_TAG = "Impacta"

MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
WEEKDAYS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]

FIELD_ORDER = ["title", "description", "date", "category", "impact", "impact_note", "sources", "tag", "keywords"]

# Blocks this script owns inside each note; everything else in the note is left alone
IMPACT_START, IMPACT_END = "{/* newsletter:impact:start */}", "{/* newsletter:impact:end */}"
RELATED_START, RELATED_END = "{/* newsletter:related:start */}", "{/* newsletter:related:end */}"
LLMS_START, LLMS_END = "<!-- newsletter:start -->", "<!-- newsletter:end -->"


# ── Frontmatter ──────────────────────────────────────────────────────────────

def _parse_value(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [v.strip().strip("'\"") for v in raw.strip("[]").split(",") if v.strip()]
    if raw.startswith('"') and raw.endswith('"'):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw[1:-1]
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1].replace("''", "'")
    if raw in ("true", "false"):
        return raw == "true"
    return raw


def split_frontmatter(text: str) -> tuple[dict, str]:
    """The notes only use flat `key: value` frontmatter, so a full YAML parser isn't needed."""
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n?", text, re.S)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        km = re.match(r"^([A-Za-z_][\w:-]*):\s*(.*)$", line)
        if km:
            meta[km.group(1)] = _parse_value(km.group(2))
    return meta, text[m.end():]


def render_frontmatter(meta: dict) -> str:
    keys = [k for k in FIELD_ORDER if k in meta] + sorted(k for k in meta if k not in FIELD_ORDER)
    lines = []
    for k in keys:
        v = meta[k]
        if v is None or v == "" or v == []:
            continue
        # JSON strings and arrays are valid YAML, and survive quotes and colons in titles
        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "---\n" + "\n".join(lines) + "\n---\n"


# ── Notes ────────────────────────────────────────────────────────────────────

@dataclass
class Note:
    slug: str
    path: Path
    meta: dict
    body: str
    day: date = field(init=False)

    def __post_init__(self):
        self.day = date.fromisoformat(str(self.meta["date"]))

    @property
    def title(self) -> str:
        return self.meta.get("title", self.slug)

    @property
    def description(self) -> str:
        return self.meta.get("description", "")

    @property
    def category(self) -> str:
        return self.meta.get("category", "economia")

    @property
    def impact(self) -> str:
        return self.meta.get("impact", "ninguno")

    @property
    def impact_note(self) -> str:
        return self.meta.get("impact_note", "")

    @property
    def page(self) -> str:
        return f"es/newsletter/notas/{self.slug}"

    @property
    def href(self) -> str:
        return f"/{self.page}"


def load_notes() -> list[Note]:
    notes = []
    for path in NOTES_DIR.glob("*.mdx"):
        meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        if not meta.get("date"):
            print(f"  skipping {path.name}: no date in frontmatter", file=sys.stderr)
            continue
        if meta.get("category") not in CATEGORY:
            meta["category"] = "economia"
        if meta.get("impact") not in IMPACTS:
            meta["impact"] = "ninguno"
        notes.append(Note(path.stem, path, meta, body))
    # Newest first; same-day ties by impact so what matters most leads
    rank = {"alto": 0, "medio": 1, "ninguno": 2}
    notes.sort(key=lambda n: (-n.day.toordinal(), rank[n.impact], n.slug))
    return notes


def write_note(slug: str, meta: dict, body: str) -> Path:
    """Used by the daily aggregator; build() adds the generated blocks afterwards."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTES_DIR / f"{slug}.mdx"
    path.write_text(render_frontmatter(meta) + "\n" + body.strip() + "\n", encoding="utf-8")
    return path


COMPONENT_TAG = re.compile(r"<(/?)([A-Z][A-Za-z]*)\b[^>]*?(/?)>")


def escape_dollars(body: str) -> str:
    """Mintlify reads two amounts like "$1 ... $300" as inline LaTeX; outside code, $ is always money."""
    parts = re.split(r"(```.*?```|`[^`\n]*`)", body, flags=re.S)
    return "".join(p if p.startswith("`") else re.sub(r"(?<!\\)\$", r"\\$", p) for p in parts)


def repair_mdx(body: str) -> str:
    """The model sometimes glues a closing tag to the last line of text, which MDX rejects."""
    body = re.sub(r"(?m)^(?!\s*<[A-Z])(.*\S)[ \t]*(</[A-Z][A-Za-z]*>)[ \t]*$", r"\1\n\2", body)
    return escape_dollars(body)


def mdx_problems(body: str) -> list[str]:
    """Unbalanced Mintlify components break the whole deploy, so they're caught before saving."""
    problems, stack = [], []
    text = re.sub(r"```.*?```", "", body, flags=re.S)
    for m in COMPONENT_TAG.finditer(text):
        closing, name, self_closing = m.group(1), m.group(2), m.group(3)
        if self_closing:
            continue
        if not closing:
            stack.append(name)
        elif not stack or stack[-1] != name:
            problems.append(f"</{name}> without matching <{name}>")
        else:
            stack.pop()
    problems += [f"<{name}> never closed" for name in stack]
    return problems


# ── Rendering helpers ────────────────────────────────────────────────────────

def t(text: str) -> str:
    """Text as a JSX string expression: MDX would otherwise parse {, <, * in titles."""
    return "{" + json.dumps(str(text), ensure_ascii=False) + "}"


def attr(text: str) -> str:
    return json.dumps(str(text), ensure_ascii=False)


def long_date(d: date) -> str:
    return f"{WEEKDAYS[d.weekday()]} {d.day} de {MONTHS[d.month - 1]} de {d.year}"


def short_date(d: date) -> str:
    return f"{d.day} {MONTHS[d.month - 1][:3]}"


def month_label(key: str) -> str:
    y, m = key.split("-")
    return f"{MONTHS[int(m) - 1].capitalize()} {y}"


def month_key(d: date) -> str:
    return f"{d.year}-{d.month:02d}"


def clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def time_tag(d: date, fmt=short_date) -> str:
    # Mintlify strips <time>, so the machine-readable date goes in a data attribute
    return f'<span className="fa-date" data-date="{d.isoformat()}">{t(fmt(d))}</span>'


def impact_badge(note: Note) -> str:
    if note.impact == "alto":
        return '<span className="fa-badge fa-badge-alto">Impacta en Facturear</span>'
    if note.impact == "medio":
        return '<span className="fa-badge fa-badge-medio">Para quienes facturan</span>'
    return ""


def kicker(note: Note, with_link: bool = False) -> str:
    label = CATEGORY[note.category][0]
    if with_link:
        return f'<a className="fa-kicker" href="/es/newsletter/temas/{note.category}">{t(label)}</a>'
    return f'<span className="fa-kicker">{t(label)}</span>'


def list_item(note: Note, show_category: bool = True) -> str:
    meta = [time_tag(note.day, lambda d: f"{d.day} de {MONTHS[d.month - 1]}")]
    if show_category:
        meta.append(t(CATEGORY[note.category][0]))
    note_line = f'\n    <p className="fa-list-note"><strong>En Facturear:</strong> {t(note.impact_note)}</p>' if note.impact != "ninguno" and note.impact_note else ""
    return f"""  <li className="fa-list-item fa-impact-{note.impact}">
    <div className="fa-list-meta">{' · '.join(meta)} {impact_badge(note)}</div>
    <a className="fa-list-title" href="{note.href}">{t(note.title)}</a>
    <p className="fa-list-dek">{t(clip(note.description, 240))}</p>{note_line}
  </li>"""


def note_list(notes: list[Note], show_category: bool = True) -> str:
    return '<ul className="not-prose fa-list">\n' + "\n".join(list_item(n, show_category) for n in notes) + "\n</ul>"


def by_month(notes: list[Note]) -> dict[str, list[Note]]:
    months: dict[str, list[Note]] = {}
    for n in notes:
        months.setdefault(month_key(n.day), []).append(n)
    return months


def write_page(path: Path, meta: dict, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = render_frontmatter(meta) + "\n{/* Generado por scripts/newsletter.py: no editar a mano */}\n\n" + body.strip() + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")


# ── Pages ────────────────────────────────────────────────────────────────────

def front_page(notes: list[Note], today: date) -> str:
    alto = [n for n in notes if n.impact == "alto"][:4]
    # The red band already carries what changes invoicing; the lead is the day's news
    lead = next((n for n in notes if n not in alto), notes[0])
    rest = [n for n in notes if n is not lead and n not in alto]
    latest = rest[:6]
    secondary = rest[6:10]
    shown = {lead.slug, *(n.slug for n in latest), *(n.slug for n in secondary), *(n.slug for n in alto)}
    vencimientos = [n for n in notes if n.category == "vencimientos" and n.slug not in shown][:5]
    shown |= {n.slug for n in vencimientos}

    nav = "\n".join(f'    <a href="/es/newsletter/temas/{k}">{t(label)}</a>' for k, label, _, _ in CATEGORIES if any(n.category == k for n in notes))

    alert_items = "\n".join(
        f"""    <a className="fa-alert-item" href="{n.href}">
      {time_tag(n.day)}
      <span className="fa-alert-title">{t(n.title)}</span>
      <span className="fa-alert-note">{t(clip(n.impact_note or n.description, 230))}</span>
    </a>"""
        for n in alto
    )

    latest_items = "\n".join(
        f"""      <li>
        {kicker(n)}
        <a className="fa-headline-sm" href="{n.href}">{t(n.title)}</a>
        {f'<p className="fa-dek-sm">{t(clip(n.description, 150))}</p>' if i < 3 else ''}
        <div className="fa-meta">{time_tag(n.day)} {impact_badge(n)}</div>
      </li>"""
        for i, n in enumerate(latest)
    )

    secondary_items = "\n".join(
        f"""        <div className="fa-second">
          {kicker(n)}
          <a className="fa-headline-sm" href="{n.href}">{t(n.title)}</a>
          <p className="fa-dek-sm">{t(clip(n.description, 170))}</p>
          <div className="fa-meta">{time_tag(n.day)} {impact_badge(n)}</div>
        </div>"""
        for n in secondary
    )

    venc_items = "\n".join(
        f"""      <li>
        <a href="{n.href}">{t(n.title)}</a>
        <div className="fa-meta">{time_tag(n.day)}</div>
      </li>"""
        for n in vencimientos
    )

    section_boxes = []
    for key, label, _, _ in CATEGORIES:
        items = [n for n in notes if n.category == key and n.slug not in shown][:4]
        if not items:
            continue
        head, *others = items
        others_html = "\n".join(f'        <li><a href="{n.href}">{t(n.title)}</a> {time_tag(n.day)}</li>' for n in others)
        section_boxes.append(f"""    <section className="fa-section">
      <h3 className="fa-section-title"><a href="/es/newsletter/temas/{key}">{t(label)}</a></h3>
      <a className="fa-headline-sm" href="{head.href}">{t(head.title)}</a>
      <p className="fa-dek-sm">{t(clip(head.description, 140))}</p>
      <div className="fa-meta">{time_tag(head.day)} {impact_badge(head)}</div>
      <ul className="fa-section-more">
{others_html}
      </ul>
    </section>""")

    months = by_month(notes)
    archive = "\n".join(f'    <a href="/es/newsletter/archivo/{k}">{t(month_label(k))} <span>{len(v)}</span></a>' for k, v in months.items())

    return f"""<div className="not-prose fa-np">
  <div className="fa-mast" role="banner">
    <div className="fa-mast-strip">
      <span>{t(long_date(today).capitalize())}</span>
      <span>{t(f"Edición N.º {len(notes)}")}</span>
      <span>Buenos Aires · Argentina</span>
    </div>
    <p className="fa-mast-name">Newsletter <em>Facturear</em></p>
    <p className="fa-mast-motto">Las novedades impositivas que importan para facturar en Argentina, todos los días.</p>
    <div className="fa-mast-nav" role="navigation" aria-label="Temas">
{nav}
    </div>
  </div>

  <section className="fa-alert" aria-label="Novedades que impactan en Facturear">
    <div className="fa-alert-head">
      <span className="fa-alert-label">Impacta en Facturear</span>
      <span className="fa-alert-sub">Cambios que modifican cómo emitís tus comprobantes</span>
      <a href="/es/newsletter/impacto-facturear">Ver todas →</a>
    </div>
    <div className="fa-alert-grid">
{alert_items}
    </div>
  </section>

  <div className="fa-front">
    <div className="fa-lead" role="article">
      {kicker(lead, with_link=True)}
      <a className="fa-headline" href="{lead.href}">{t(lead.title)}</a>
      <p className="fa-dek">{t(lead.description)}</p>
      {f'<p className="fa-lead-impact"><strong>Qué cambia en Facturear:</strong> {t(lead.impact_note)}</p>' if lead.impact != "ninguno" and lead.impact_note else ''}
      <div className="fa-meta">{time_tag(lead.day, long_date)} {impact_badge(lead)}</div>
      <a className="fa-readmore" href="{lead.href}">Leer la nota completa →</a>
      <div className="fa-seconds">
{secondary_items}
      </div>
    </div>

    <div className="fa-col fa-latest">
      <h2 className="fa-colhead">Últimas noticias</h2>
      <ul>
{latest_items}
      </ul>
    </div>

    <div className="fa-col fa-aside" role="complementary">
      <h2 className="fa-colhead">Vencimientos y prórrogas</h2>
      <ul className="fa-venc">
{venc_items}
      </ul>
      <div className="fa-sub" data-fa-newsletter="inline">
        <p className="fa-sub-title">Recibí el newsletter</p>
        <p className="fa-sub-text">Todos los lunes a la mañana, lo que cambió en la semana para facturar. Sin spam, te das de baja con un clic.</p>
      </div>
    </div>
  </div>

  <div className="fa-sections">
{chr(10).join(section_boxes)}
  </div>

  <div className="fa-foot" role="contentinfo">
    <h2 className="fa-colhead">Hemeroteca</h2>
    <div className="fa-archive" role="navigation" aria-label="Archivo por mes">
{archive}
    </div>
    <p className="fa-foot-note">Las notas se escriben a partir de lo publicado por los principales medios económicos del país y las resoluciones de ARCA. No reemplazan el asesoramiento de tu contador.</p>
  </div>
</div>"""


def impact_page(notes: list[Note]) -> str:
    alto = [n for n in notes if n.impact == "alto"]
    medio = [n for n in notes if n.impact == "medio"]
    return f"""Las novedades que cambian **cómo se emiten los comprobantes electrónicos**: nuevos obligados, datos que pasan a ser requeridos, cambios en CAE y CAEA, tipos de comprobante y fechas límite para adaptar sistemas. Cuando algo de esto sale, Facturear lo incorpora a la plataforma y a la API.

<Tip>
Si integrás por API, suscribite al newsletter: te avisamos antes de que un cambio de ARCA llegue a producción.
</Tip>

## Impacta en Facturear

{note_list(alto)}

## Para quienes facturan

Novedades que no cambian la plataforma, pero sí decisiones de facturación: topes del Monotributo, recategorizaciones, percepciones y controles sobre comprobantes.

{note_list(medio)}
"""


def topic_page(key: str, notes: list[Note]) -> str:
    label, _, guide = CATEGORY[key]
    alto = [n for n in notes if n.impact == "alto"]
    parts = [f"{len(notes)} notas, de la más reciente a la más antigua."]
    if guide:
        parts.append(f"\n<Card title={attr(f'Guía de {label}')} icon=\"book-open\" href=\"{guide}\">\nLo básico, explicado sin apuro, en la sección Impuestos.\n</Card>")
    if alto:
        parts.append(f"\n## Impacta en Facturear\n\n{note_list(alto, show_category=False)}")
    for k, items in by_month(notes).items():
        parts.append(f"\n## {month_label(k)}\n\n{note_list(items, show_category=False)}")
    return "\n".join(parts)


def month_page(key: str, notes: list[Note], prev_key: str | None, next_key: str | None) -> str:
    counts = {}
    for n in notes:
        counts[n.category] = counts.get(n.category, 0) + 1
    summary = ", ".join(f"{CATEGORY[k][0]} ({c})" for k, c in sorted(counts.items(), key=lambda kv: -kv[1]))
    alto = sum(1 for n in notes if n.impact == "alto")
    nav = []
    if next_key:
        nav.append(f'<a href="/es/newsletter/archivo/{next_key}">← {t(month_label(next_key))}</a>')
    if prev_key:
        nav.append(f'<a href="/es/newsletter/archivo/{prev_key}">{t(month_label(prev_key))} →</a>')
    nav_html = f'\n\n<div className="not-prose fa-pager">{" ".join(nav)}</div>' if nav else ""
    impact_line = f" {alto} {'impacta' if alto == 1 else 'impactan'} directamente en la facturación." if alto else ""
    return f"""{len(notes)} notas publicadas en {month_label(key).lower()}.{impact_line} Por tema: {summary}.

{note_list(notes)}{nav_html}"""


# ── Blocks inside each note ──────────────────────────────────────────────────

def strip_block(body: str, start: str, end: str) -> str:
    return re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n*", "", body, flags=re.S)


def impact_block(note: Note) -> str:
    if note.impact == "ninguno" or not note.impact_note:
        return ""
    component, title = ("Warning", "Impacta en Facturear") if note.impact == "alto" else ("Info", "Para quienes facturan")
    return f"{IMPACT_START}\n<{component} title={attr(title)}>\n{t(note.impact_note)} [Ver todas las novedades que impactan](/es/newsletter/impacto-facturear).\n</{component}>\n{IMPACT_END}\n\n"


def related_block(note: Note, notes: list[Note]) -> str:
    label = CATEGORY[note.category][0]
    same = [n for n in notes if n.category == note.category and n.slug != note.slug]
    # Closest in time first: what readers of this note most likely want next
    same.sort(key=lambda n: (abs((n.day - note.day).days), n.slug))
    items = "\n".join(f'    <li><a href="{n.href}">{t(n.title)}</a> {time_tag(n.day)}</li>' for n in same[:5])
    if not items:
        return ""
    return f"""

{RELATED_START}
<div className="not-prose fa-related">
  <p className="fa-related-head">{t(f"Más sobre {label}")}</p>
  <ul>
{items}
  </ul>
  <div className="fa-related-links">
    <a href="/es/newsletter/temas/{note.category}">{t(f"Todas las notas de {label}")} →</a>
    <a href="/es/newsletter/archivo/{month_key(note.day)}">{t(f"Archivo de {month_label(month_key(note.day)).lower()}")} →</a>
  </div>
  <div data-fa-newsletter="inline" className="fa-sub fa-sub-compact">
    <p className="fa-sub-title">¿Te sirvió? Recibí el resumen de los lunes</p>
  </div>
</div>
{RELATED_END}
"""


def update_note(note: Note, notes: list[Note]) -> None:
    meta = dict(note.meta)
    if note.impact == "alto":
        meta["tag"] = IMPACT_TAG
    elif meta.get("tag") == IMPACT_TAG:
        meta.pop("tag")
    body = strip_block(strip_block(note.body, IMPACT_START, IMPACT_END), RELATED_START, RELATED_END).strip()
    content = render_frontmatter(meta) + "\n" + impact_block(note) + body + related_block(note, notes)
    if note.path.read_text(encoding="utf-8") != content:
        note.path.write_text(content, encoding="utf-8")


# ── Navigation, llms.txt ─────────────────────────────────────────────────────

def navigation(notes: list[Note]) -> dict:
    topics = [f"es/newsletter/temas/{k}" for k, *_ in CATEGORIES if any(n.category == k for n in notes)]
    months = [
        {"group": month_label(k), "pages": [f"es/newsletter/archivo/{k}", *(n.page for n in items)]}
        for k, items in by_month(notes).items()
    ]
    return {
        "tab": TAB_NAME,
        "groups": [
            {"group": "Newsletter Facturear", "pages": ["es/newsletter/index", "es/newsletter/impacto-facturear"]},
            {"group": "Temas", "pages": topics},
            {"group": "Archivo", "pages": months},
        ],
    }


def update_docs_json(notes: list[Note]) -> None:
    docs = json.loads(DOCS_JSON.read_text(encoding="utf-8"))
    tabs = docs["navigation"]["tabs"]
    tab = navigation(notes)
    idx = next((i for i, tb in enumerate(tabs) if tb.get("tab") == TAB_NAME), None)
    if idx is None:
        tabs.insert(0, tab)
    else:
        tabs[idx] = tab
    DOCS_JSON.write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_feed(notes: list[Note]) -> None:
    """Latest notes as JSON for the weekly email sent from facture.ar."""
    feed = {
        "version": 1,
        "url": f"{SITE}/es/newsletter/index",
        "notes": [
            {
                "slug": n.slug,
                "title": n.title,
                "description": n.description,
                "date": n.day.isoformat(),
                "category": n.category,
                "category_label": CATEGORY[n.category][0],
                "impact": n.impact,
                "impact_note": n.impact_note or None,
                "url": f"{SITE}{n.href}",
            }
            for n in notes[:FEED_SIZE]
        ],
    }
    content = json.dumps(feed, ensure_ascii=False, indent=1) + "\n"
    if not FEED_JSON.exists() or FEED_JSON.read_text(encoding="utf-8") != content:
        FEED_JSON.write_text(content, encoding="utf-8")


def update_llms_txt(notes: list[Note]) -> None:
    if not LLMS_TXT.exists():
        return
    lines = [
        LLMS_START,
        "## Newsletter Facturear (novedades impositivas)",
        "",
        f"- [Portada del newsletter]({SITE}/es/newsletter/index): Novedades impositivas argentinas, actualizadas a diario.",
        f"- [Novedades que impactan en Facturear]({SITE}/es/newsletter/impacto-facturear): Cambios de ARCA que modifican cómo se emiten comprobantes electrónicos.",
        *(f"- [{label}]({SITE}/es/newsletter/temas/{k}): {desc}" for k, label, desc, _ in CATEGORIES if any(n.category == k for n in notes)),
        "",
        "### Últimas notas",
        "",
        *(f"- [{n.title}]({SITE}{n.href}) ({n.day.isoformat()}): {clip(n.description, 180)}" for n in notes[:25]),
        LLMS_END,
    ]
    block = "\n".join(lines)
    text = LLMS_TXT.read_text(encoding="utf-8")
    if LLMS_START in text:
        text = re.sub(re.escape(LLMS_START) + r".*?" + re.escape(LLMS_END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    LLMS_TXT.write_text(text, encoding="utf-8")


# ── Build ────────────────────────────────────────────────────────────────────

def build(today: date | None = None) -> None:
    today = today or datetime.now(tz=ART).date()
    notes = load_notes()
    if not notes:
        print("No notes found; nothing to build.", file=sys.stderr)
        return

    for note in notes:
        update_note(note, notes)

    write_page(
        NEWSLETTER_DIR / "index.mdx",
        {
            "title": "Newsletter Facturear: novedades impositivas de Argentina",
            "sidebarTitle": "Portada",
            "description": "El diario impositivo de Facturear: ARCA, Monotributo, IVA, Ganancias, Ingresos Brutos y vencimientos, con foco en lo que cambia para facturar.",
            "mode": "custom",
            "keywords": ["novedades impositivas", "ARCA", "AFIP", "monotributo", "factura electrónica", "vencimientos"],
        },
        front_page(notes, today),
    )
    write_page(
        NEWSLETTER_DIR / "impacto-facturear.mdx",
        {
            "title": "Novedades que impactan en Facturear",
            "sidebarTitle": "Impacta en Facturear",
            "description": "Cambios de ARCA que modifican cómo se emiten facturas y comprobantes electrónicos en Argentina, y cómo los resuelve Facturear.",
            "icon": "triangle-exclamation",
        },
        impact_page(notes),
    )

    topics_dir = NEWSLETTER_DIR / "temas"
    live_topics = set()
    for key, label, desc, _ in CATEGORIES:
        items = [n for n in notes if n.category == key]
        if not items:
            continue
        live_topics.add(f"{key}.mdx")
        write_page(topics_dir / f"{key}.mdx", {"title": f"{label}: novedades", "sidebarTitle": label, "description": desc}, topic_page(key, items))

    archive_dir = NEWSLETTER_DIR / "archivo"
    months = by_month(notes)
    keys = list(months)
    for i, key in enumerate(keys):
        write_page(
            archive_dir / f"{key}.mdx",
            {"title": f"Novedades impositivas de {month_label(key).lower()}", "sidebarTitle": f"Todo {month_label(key).lower()}", "description": f"Todas las novedades impositivas de Argentina publicadas en {month_label(key).lower()} en el Newsletter Facturear."},
            month_page(key, months[key], keys[i + 1] if i + 1 < len(keys) else None, keys[i - 1] if i > 0 else None),
        )

    # Pages for topics or months that no longer have notes would 404 from the nav
    for stale in [*(p for p in topics_dir.glob("*.mdx") if p.name not in live_topics), *(p for p in archive_dir.glob("*.mdx") if p.stem not in months)]:
        stale.unlink()

    update_docs_json(notes)
    update_llms_txt(notes)
    write_feed(notes)
    print(f"Newsletter built: {len(notes)} notes, {len(live_topics)} topics, {len(months)} months.")


if __name__ == "__main__":
    build()
