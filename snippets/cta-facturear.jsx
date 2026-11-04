/**
 * One call to action per tax page, pointing to www.facture.ar with utm_source=docs,
 * utm_medium=referral, utm_campaign=<page> and utm_content=<tema>: the app keeps that first touch
 * and shows the signup under the "Docs" channel in /admin, split by page.
 *
 * <CtaFacturear page="iva-alicuotas" tema="iva" />
 * tema: monotributo | iva | cuit | exterior | general (default)
 *
 * The newsletter notes get the same markup from workers/newsletter-news/newsletter.py in facturear.
 */
export const CtaFacturear = ({ page, tema = 'general' }) => {
  const ctas = {
    monotributo: {
      title: '¿En qué categoría de Monotributo estás?',
      text: 'Cargá lo que facturaste en los últimos 12 meses y te decimos la categoría y la cuota que te tocan. Gratis y sin crear cuenta.',
      label: 'Calcular mi categoría',
      path: '/herramientas/calculadora-categoria-monotributo',
    },
    iva: {
      title: 'Calculá el IVA de un precio',
      text: 'Sumale o sacale el IVA a un precio al 21%, 10,5% o 27% y mirá cómo queda en una Factura A y en una B.',
      label: 'Abrir la calculadora de IVA',
      path: '/herramientas/calculadora-iva',
    },
    cuit: {
      title: '¿Ese CUIT está bien escrito?',
      text: 'Validá el dígito verificador de un CUIT o CUIL antes de cargarlo en una factura.',
      label: 'Validar un CUIT',
      path: '/herramientas/validar-cuit',
    },
    exterior: {
      title: 'Facturá al exterior sin entrar a ARCA',
      text: 'Emití Factura E en dólares, euros o pesos con la cotización del día. La exportación en ARCA la habilitamos nosotros.',
      label: 'Ver cómo funciona',
      path: '/lp/factura-exterior',
    },
    general: {
      title: 'Facturá en ARCA sin entrar a ARCA',
      text: 'Conectás tu CUIT en 5 minutos y facturás desde la web, por WhatsApp o con un Excel. Empezás con 14 días del plan Pro, sin tarjeta.',
      label: 'Crear cuenta gratis',
      path: '/auth/signin',
    },
  };
  const cta = ctas[tema] || ctas.general;
  const link = path => {
    const params = new URLSearchParams({ utm_source: 'docs', utm_medium: 'referral', utm_campaign: page || 'docs', utm_content: tema });
    return `https://www.facture.ar${path}?${params.toString()}`;
  };
  const isSignup = cta.path === ctas.general.path;
  return (
    <div className="not-prose fa-cta" data-fa-cta={tema}>
      <p className="fa-cta-title">{cta.title}</p>
      <p className="fa-cta-text">{cta.text}</p>
      <div className="fa-cta-actions">
        <a className="fa-cta-button" href={link(cta.path)}>
          {cta.label} →
        </a>
        {isSignup ? null : (
          <a className="fa-cta-link" href={link(ctas.general.path)}>
            o creá tu cuenta gratis
          </a>
        )}
      </div>
    </div>
  );
};
