// Standalone design tool (app.html): the same optimizer panel as the walkthrough page.
import { OptimizerPanel, SCENARIOS } from './panel.js?v=a7115a82';

const start = new URLSearchParams(location.search).get('scenario');
const pg = new OptimizerPanel(document.getElementById('playground'), { views: SCENARIOS.sunmoon.views, editable: true });
if (start && start !== 'sunmoon' && SCENARIOS[start]) {
  document.querySelector('#playground [data-r=preset]').value = start;
  pg.applyScenario(start);
}
window.__app = { pg };
