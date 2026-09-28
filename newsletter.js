/*
 * Newsletter Facturear: subscribe prompts for docs.facture.ar.
 *
 * - Modal: once someone has read a second news note, it offers the newsletter a single time.
 *   Closing it is remembered in localStorage and it never shows again.
 * - Banner: a small pill in the corner on every page, until they subscribe. Its × hides it
 *   for the rest of the session only.
 * - Inline: any element with data-fa-newsletter="inline" (front page, end of each note)
 *   gets the form mounted inside.
 *
 * Mintlify loads every .js in the repo on every page and navigates client-side, so this
 * watches the path instead of relying on page loads.
 */
(function () {
  if (typeof window === 'undefined' || window.__faNewsletter) return;
  window.__faNewsletter = true;

  var API = 'https://www.facture.ar/api/newsletter/subscribe';
  var STATE_KEY = 'fa-newsletter';
  var VIEWS_KEY = 'fa-newsletter-views';
  var BANNER_KEY = 'fa-newsletter-banner-hidden';
  var NOTE_PATH = /^\/es\/newsletter\/notas\/([^/?#]+)/;
  var VIEWS_BEFORE_MODAL = 2;
  var MODAL_DELAY_MS = 6000;

  // Storage can throw (private mode, blocked site data): every access degrades to "no value"
  function read(store, key) {
    try { return window[store].getItem(key); } catch (e) { return null; }
  }
  function write(store, key, value) {
    try { window[store].setItem(key, value); } catch (e) { /* not persisted, that's fine */ }
  }
  function state() {
    try { return (JSON.parse(read('localStorage', STATE_KEY) || '{}') || {}).state || null; } catch (e) { return null; }
  }
  function setState(value) {
    write('localStorage', STATE_KEY, JSON.stringify({ state: value, at: new Date().toISOString() }));
  }
  function views() {
    try { return JSON.parse(read('localStorage', VIEWS_KEY) || '[]') || []; } catch (e) { return []; }
  }

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (k === 'text') node.textContent = attrs[k];
      else if (k === 'on') Object.keys(attrs.on).forEach(function (ev) { node.addEventListener(ev, attrs.on[ev]); });
      else node.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(c); });
    return node;
  }

  function form(source, onDone) {
    var input = el('input', { type: 'email', name: 'email', required: 'required', placeholder: 'tu@email.com', autocomplete: 'email', 'aria-label': 'Tu email' });
    // Bots fill every field; people never see this one
    var honeypot = el('input', { type: 'text', name: 'website', tabindex: '-1', autocomplete: 'off', 'aria-hidden': 'true', class: 'fa-hp' });
    var button = el('button', { type: 'submit', text: 'Suscribirme' });
    var msg = el('p', { class: 'fa-form-msg', role: 'status', 'aria-live': 'polite' });
    var f = el('form', { class: 'fa-form', novalidate: 'novalidate' }, [input, honeypot, button, msg]);

    f.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var email = (input.value || '').trim();
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(email)) {
        msg.className = 'fa-form-msg fa-err';
        msg.textContent = 'Revisá el email, parece incompleto.';
        input.focus();
        return;
      }
      button.disabled = true;
      msg.className = 'fa-form-msg';
      msg.textContent = 'Enviando…';
      fetch(API, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email: email, source: source, path: location.pathname, website: honeypot.value }),
      })
        .then(function (res) {
          return res.json().catch(function () { return {}; }).then(function (body) { return { ok: res.ok, body: body }; });
        })
        .then(function (r) {
          if (!r.ok) throw new Error((r.body && r.body.error) || 'error');
          setState('subscribed');
          msg.className = 'fa-form-msg fa-ok';
          msg.textContent = r.body.status === 'active' ? 'Ya estabas suscripto. ¡Gracias!' : 'Listo. Te mandamos un mail para confirmar la suscripción.';
          input.disabled = true;
          removeBanner();
          if (onDone) setTimeout(onDone, 2200);
        })
        .catch(function (err) {
          button.disabled = false;
          msg.className = 'fa-form-msg fa-err';
          msg.textContent = err && err.message && err.message !== 'error' && err.message.length < 140 ? err.message : 'No pudimos suscribirte. Probá de nuevo en un rato.';
        });
    });
    return f;
  }

  // ── Modal ────────────────────────────────────────────────────────────────
  var modal = null;
  var modalTimer = null;

  function closeModal(remember) {
    if (!modal) return;
    if (remember && state() !== 'subscribed') setState('dismissed');
    modal.remove();
    modal = null;
    document.removeEventListener('keydown', onKey);
    renderBanner();
  }
  function onKey(ev) { if (ev.key === 'Escape') closeModal(true); }

  function openModal(source) {
    if (modal || !document.body) return;
    removeBanner();
    var box = el('div', { class: 'fa-modal', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'fa-modal-title' }, [
      el('button', { class: 'fa-modal-close', type: 'button', 'aria-label': 'Cerrar', text: '×', on: { click: function () { closeModal(true); } } }),
      el('div', { class: 'fa-modal-kicker', text: 'Newsletter Facturear' }),
      el('h2', { class: 'fa-modal-title', id: 'fa-modal-title', text: 'Las novedades impositivas, en tu mail' }),
      el('p', { class: 'fa-modal-text', text: 'Te avisamos lo que cambia para facturar antes de que te afecte. Leés en dos minutos y seguís con lo tuyo.' }),
      el('ul', { class: 'fa-modal-points' }, [
        el('li', { text: 'Cambios de ARCA que impactan en tus comprobantes' }),
        el('li', { text: 'Vencimientos y topes del Monotributo' }),
        el('li', { text: 'Sin spam: te das de baja con un clic' }),
      ]),
      form(source, function () { closeModal(false); }),
      el('button', { class: 'fa-modal-later', type: 'button', text: 'No, gracias', on: { click: function () { closeModal(true); } } }),
    ]);
    modal = el('div', { class: 'fa-modal-backdrop', on: { click: function (ev) { if (ev.target === modal) closeModal(true); } } }, [box]);
    document.body.appendChild(modal);
    document.addEventListener('keydown', onKey);
    var input = box.querySelector('input[type=email]');
    if (input) setTimeout(function () { input.focus(); }, 50);
  }

  // ── Banner ───────────────────────────────────────────────────────────────
  var banner = null;

  function removeBanner() {
    if (banner) { banner.remove(); banner = null; }
  }
  function bannerAllowed() {
    return state() !== 'subscribed' && read('sessionStorage', BANNER_KEY) !== '1' && !/^\/es\/api-reference/.test(location.pathname);
  }
  function renderBanner() {
    if (!bannerAllowed() || modal) return removeBanner();
    if (banner || !document.body) return;
    banner = el('div', { class: 'fa-banner', role: 'complementary', 'aria-label': 'Suscripción al newsletter' }, [
      el('button', { class: 'fa-banner-open', type: 'button', text: 'Recibí las novedades impositivas', on: { click: function () { openModal('banner'); } } }),
      el('button', {
        class: 'fa-banner-x', type: 'button', 'aria-label': 'Ocultar', text: '×',
        on: { click: function () { write('sessionStorage', BANNER_KEY, '1'); removeBanner(); } },
      }),
    ]);
    document.body.appendChild(banner);
  }

  // ── Inline boxes ─────────────────────────────────────────────────────────
  function mountInline() {
    var nodes = document.querySelectorAll('[data-fa-newsletter="inline"]:not([data-fa-mounted])');
    Array.prototype.forEach.call(nodes, function (node) {
      node.setAttribute('data-fa-mounted', '1');
      if (state() === 'subscribed') {
        node.appendChild(el('p', { class: 'fa-form-msg fa-ok', text: 'Ya estás suscripto al newsletter. ¡Gracias!' }));
        return;
      }
      node.appendChild(form(NOTE_PATH.test(location.pathname) ? 'note' : 'front-page'));
      node.appendChild(el('p', { class: 'fa-fine', text: 'Te pedimos confirmar por mail. Podés darte de baja cuando quieras.' }));
    });
  }

  // ── Route handling ───────────────────────────────────────────────────────
  var lastPath = null;

  function onRoute() {
    var path = location.pathname;
    mountInline();
    if (path === lastPath) return;
    lastPath = path;
    clearTimeout(modalTimer);
    renderBanner();

    var m = path.match(NOTE_PATH);
    if (!m) return;
    var seen = views();
    if (seen.indexOf(m[1]) === -1) {
      seen.push(m[1]);
      write('localStorage', VIEWS_KEY, JSON.stringify(seen.slice(-50)));
    }
    if (!state() && seen.length >= VIEWS_BEFORE_MODAL) {
      modalTimer = setTimeout(function () {
        if (!state() && location.pathname === path) openModal('modal');
      }, MODAL_DELAY_MS);
    }
  }

  ['pushState', 'replaceState'].forEach(function (fn) {
    var original = history[fn];
    history[fn] = function () {
      var result = original.apply(this, arguments);
      setTimeout(onRoute, 0);
      return result;
    };
  });
  window.addEventListener('popstate', onRoute);
  // Page content re-renders after navigation, so inline boxes can appear late
  setInterval(onRoute, 800);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', onRoute);
  else onRoute();
})();
