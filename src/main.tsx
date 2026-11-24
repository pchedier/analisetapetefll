import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import {ErrorBoundary} from './components/ErrorBoundary.tsx';
import './index.css';

if (typeof navigator !== 'undefined' && 'serviceWorker' in navigator) {
  let refreshing = false;
  const hadController = Boolean(navigator.serviceWorker.controller);
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!hadController || refreshing) return;
    refreshing = true;
    window.location.reload();
  });
  navigator.serviceWorker.getRegistrations().then((regs) => {
    regs.forEach((r) => {
      r.update().catch(() => {});
    });
  }).catch(() => {});
}

createRoot(document.getElementById('root')!).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
