import React, { useState, useEffect } from 'react';
import {
  Key,
  ExternalLink,
  Check,
  AlertCircle,
  Sparkles,
  Eye,
  EyeOff,
  Trash2,
  Loader2,
  CheckCircle2,
  Cpu,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import {
  AI_PROVIDER_PREF_KEY,
  AiProviderPref,
  GROQ_API_KEY_STORAGE_KEY,
  OPENROUTER_API_KEY_STORAGE_KEY,
  PREFERRED_GEMINI_MODEL_KEY,
  SELECTABLE_GEMINI_MODELS,
  getSavedProviderConfig,
  hasAnyConfiguredAiProvider,
  hasBackendServer,
  validateGeminiKeyDirect,
  validateGroqKeyDirect,
  validateOpenRouterKeyDirect,
} from '../services/clientAiEngine';

interface ApiSettingsViewProps {
  isAiReady?: boolean;
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
  onApiKeySaved?: (key: string, isValid: boolean) => void;
}

export const API_KEY_STORAGE_KEY = 'francodroid_user_gemini_api_key';

export const ApiSettingsView: React.FC<ApiSettingsViewProps> = ({
  isAiReady = false,
  onShowToast,
  onApiKeySaved,
}) => {
  const [apiKey, setApiKey] = useState('');
  const [savedKey, setSavedKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [validationStatus, setValidationStatus] = useState<'idle' | 'valid' | 'invalid'>(
    isAiReady ? 'valid' : 'idle'
  );

  // Alternative providers & model preference
  const [groqKey, setGroqKey] = useState('');
  const [savedGroqKey, setSavedGroqKey] = useState('');
  const [showGroqKey, setShowGroqKey] = useState(false);
  const [groqStatus, setGroqStatus] = useState<'idle' | 'valid' | 'invalid'>('idle');
  const [isValidatingGroq, setIsValidatingGroq] = useState(false);

  const [openRouterKey, setOpenRouterKey] = useState('');
  const [savedOpenRouterKey, setSavedOpenRouterKey] = useState('');
  const [showOpenRouterKey, setShowOpenRouterKey] = useState(false);
  const [openRouterStatus, setOpenRouterStatus] = useState<'idle' | 'valid' | 'invalid'>('idle');
  const [isValidatingOpenRouter, setIsValidatingOpenRouter] = useState(false);

  const [providerPref, setProviderPref] = useState<AiProviderPref>('auto');
  const [preferredGeminiModel, setPreferredGeminiModel] = useState<string>('auto');

  useEffect(() => {
    try {
      const cfg = getSavedProviderConfig();
      if (cfg.geminiKey) {
        setSavedKey(cfg.geminiKey);
        setApiKey(cfg.geminiKey);
        if (isAiReady) {
          setValidationStatus('valid');
        } else {
          void validateKeyOnServer(cfg.geminiKey);
        }
      }
      if (cfg.groqKey) {
        setGroqKey(cfg.groqKey);
        setSavedGroqKey(cfg.groqKey);
        setGroqStatus('valid');
      }
      if (cfg.openRouterKey) {
        setOpenRouterKey(cfg.openRouterKey);
        setSavedOpenRouterKey(cfg.openRouterKey);
        setOpenRouterStatus('valid');
      }
      setProviderPref(cfg.providerPref);
      setPreferredGeminiModel(cfg.preferredGeminiModel || 'auto');
    } catch {
      // localStorage fallback
    }
  }, [isAiReady]);

  const validateKeyOnServer = async (
    keyToValidate: string
  ): Promise<{ valid: boolean; message?: string; quotaWarning?: boolean }> => {
    if (!keyToValidate.trim()) {
      setValidationStatus('idle');
      return { valid: false };
    }

    setIsValidating(true);
    try {
      let data: { valid: boolean; message?: string; quotaWarning?: boolean };
      if (await hasBackendServer()) {
        const res = await fetch('./api/ai/validate-key', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ apiKey: keyToValidate.trim() }),
        });
        data = await res.json();
        if (!res.ok && !data?.valid) {
          setValidationStatus('invalid');
          onApiKeySaved?.(keyToValidate.trim(), hasAnyConfiguredAiProvider());
          return { valid: false, message: data?.message };
        }
      } else {
        data = await validateGeminiKeyDirect(keyToValidate.trim());
      }

      if (data.valid) {
        setValidationStatus('valid');
        onApiKeySaved?.(keyToValidate.trim(), true);
        return { valid: true, message: data.message, quotaWarning: Boolean(data.quotaWarning) };
      } else {
        setValidationStatus('invalid');
        onApiKeySaved?.(keyToValidate.trim(), hasAnyConfiguredAiProvider());
        return { valid: false, message: data.message };
      }
    } catch {
      setValidationStatus('invalid');
      onApiKeySaved?.(keyToValidate.trim(), hasAnyConfiguredAiProvider());
      return { valid: false, message: 'Erro de rede ao validar chave.' };
    } finally {
      setIsValidating(false);
    }
  };

  const handleSave = async () => {
    const trimmed = apiKey.trim();
    if (!trimmed) {
      handleClear();
      return;
    }

    setIsValidating(true);
    onShowToast('Testando conexão com a API Gemini...', 'info');

    const validation = await validateKeyOnServer(trimmed);

    if (validation.valid) {
      try {
        localStorage.setItem(API_KEY_STORAGE_KEY, trimmed);
        setSavedKey(trimmed);
        onApiKeySaved?.(trimmed, true);
        onShowToast(
          validation.message || 'Chave Gemini válida e conectada com sucesso!',
          validation.quotaWarning ? 'info' : 'success'
        );
      } catch {
        onShowToast('Chave válida, mas falha ao gravar no navegador.', 'error');
      }
    } else {
      onShowToast(
        validation.message || 'Chave inválida. Verifique se copiou a chave inteira do Google AI Studio.',
        'error'
      );
    }
  };

  const handleClear = () => {
    setApiKey('');
    setSavedKey('');
    setValidationStatus('idle');
    try {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
      onShowToast('Chave Gemini removida.', 'info');
      onApiKeySaved?.('', hasAnyConfiguredAiProvider());
    } catch {}
  };

  const handleSaveGroq = async () => {
    const trimmed = groqKey.trim();
    if (!trimmed) {
      setGroqKey('');
      setSavedGroqKey('');
      setGroqStatus('idle');
      localStorage.removeItem(GROQ_API_KEY_STORAGE_KEY);
      onShowToast('Chave Groq removida.', 'info');
      onApiKeySaved?.(savedKey, hasAnyConfiguredAiProvider());
      return;
    }

    setIsValidatingGroq(true);
    try {
      const check = await validateGroqKeyDirect(trimmed);
      if (check.valid) {
        localStorage.setItem(GROQ_API_KEY_STORAGE_KEY, trimmed);
        setSavedGroqKey(trimmed);
        setGroqStatus('valid');
        onApiKeySaved?.(trimmed, true);
        onShowToast(check.message, 'success');
      } else {
        setGroqStatus('invalid');
        onShowToast(check.message, 'error');
      }
    } finally {
      setIsValidatingGroq(false);
    }
  };

  const handleSaveOpenRouter = async () => {
    const trimmed = openRouterKey.trim();
    if (!trimmed) {
      setOpenRouterKey('');
      setSavedOpenRouterKey('');
      setOpenRouterStatus('idle');
      localStorage.removeItem(OPENROUTER_API_KEY_STORAGE_KEY);
      onShowToast('Chave OpenRouter removida.', 'info');
      onApiKeySaved?.(savedKey, hasAnyConfiguredAiProvider());
      return;
    }

    setIsValidatingOpenRouter(true);
    try {
      const check = await validateOpenRouterKeyDirect(trimmed);
      if (check.valid) {
        localStorage.setItem(OPENROUTER_API_KEY_STORAGE_KEY, trimmed);
        setSavedOpenRouterKey(trimmed);
        setOpenRouterStatus('valid');
        onApiKeySaved?.(trimmed, true);
        onShowToast(check.message, 'success');
      } else {
        setOpenRouterStatus('invalid');
        onShowToast(check.message, 'error');
      }
    } finally {
      setIsValidatingOpenRouter(false);
    }
  };

  const handleProviderPrefChange = (nextPref: AiProviderPref) => {
    setProviderPref(nextPref);
    try {
      localStorage.setItem(AI_PROVIDER_PREF_KEY, nextPref);
      onShowToast('Ordem de provedores atualizada.', 'info');
    } catch {}
  };

  const handlePreferredModelChange = (modelId: string) => {
    setPreferredGeminiModel(modelId);
    try {
      localStorage.setItem(PREFERRED_GEMINI_MODEL_KEY, modelId);
      onShowToast('Preferência de modelo Gemini atualizada.', 'info');
    } catch {}
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 fd-fade-up">
      {/* Clean Header */}
      <div className="text-center space-y-2 py-1">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-white text-balance">
          Conectar Inteligência Artificial
        </h1>
        <p className="text-slate-400 text-sm max-w-lg mx-auto leading-relaxed">
          Configure o Google Gemini ou adicione provedores alternativos gratuitos (Groq / OpenRouter) para nunca travar com erro 503.
        </p>
      </div>

      {/* Execution Strategy & Model Preference */}
      <div className="fd-panel rounded-3xl p-5 sm:p-6 space-y-4">
        <div className="flex items-center gap-2.5 text-white font-semibold text-sm">
          <Cpu className="w-4 h-4 text-cyan-300" />
          <span>Estratégia de Execução e Modelo Anti-503</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="space-y-1.5">
            <label className="text-slate-300 font-medium block">Ordem de Provedores:</label>
            <select
              value={providerPref}
              onChange={(e) => handleProviderPrefChange(e.target.value as AiProviderPref)}
              className="w-full bg-black/50 border border-white/15 rounded-xl px-3.5 py-2.5 text-xs text-slate-100 focus:outline-none focus:border-cyan-400"
            >
              <option value="auto">Automático (Gemini → Groq → OpenRouter)</option>
              <option value="groq">Sempre usar Groq Vision primeiro</option>
              <option value="openrouter">Sempre usar OpenRouter Free primeiro</option>
              <option value="gemini">Somente Google Gemini</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-slate-300 font-medium block">Modelo Gemini Prioritário:</label>
            <select
              value={preferredGeminiModel}
              onChange={(e) => handlePreferredModelChange(e.target.value)}
              className="w-full bg-black/50 border border-white/15 rounded-xl px-3.5 py-2.5 text-xs text-slate-100 focus:outline-none focus:border-cyan-400"
            >
              {SELECTABLE_GEMINI_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Gemini API Key Input Card */}
      <div className="fd-panel rounded-3xl p-6 sm:p-7 space-y-5">
        <div className="flex items-center justify-between border-b border-white/10 pb-3.5">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-500/15 border border-cyan-500/25 flex items-center justify-center">
              <Key className="w-4 h-4 text-cyan-300" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">Chave da API Google Gemini</h3>
              <span className="text-xs text-slate-400">Suporta vídeo direto e fotos do gabarito</span>
            </div>
          </div>

          <div className="text-xs font-mono">
            {isValidating ? (
              <span className="text-cyan-300 flex items-center gap-1.5">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Testando...
              </span>
            ) : validationStatus === 'valid' ? (
              <span className="text-emerald-300 flex items-center gap-1.5 font-semibold">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Conectada
              </span>
            ) : validationStatus === 'invalid' ? (
              <span className="text-rose-300 flex items-center gap-1.5 font-semibold">
                <AlertCircle className="w-4 h-4 text-rose-400" />
                Inválida
              </span>
            ) : (
              <span className="text-slate-400">Não configurada</span>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs text-slate-300 font-medium block">
            Cole sua chave do Google AI Studio:
          </label>
          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => {
                setApiKey(e.target.value);
                setValidationStatus('idle');
              }}
              placeholder="Cole aqui sua chave do Google AI Studio..."
              className="w-full bg-black/50 border border-white/15 rounded-xl pl-4 pr-12 py-3 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400 font-mono transition-colors"
            />
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 text-slate-400 hover:text-white transition-colors"
              aria-label={showKey ? 'Ocultar chave' : 'Mostrar chave'}
            >
              {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between pt-2">
          {savedKey ? (
            <button
              onClick={handleClear}
              className="text-xs text-slate-400 hover:text-rose-400 flex items-center gap-1.5 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Remover Chave</span>
            </button>
          ) : (
            <div />
          )}

          <button
            onClick={handleSave}
            disabled={isValidating || !apiKey.trim()}
            className={`px-6 py-2.5 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
              isValidating || !apiKey.trim()
                ? 'bg-white/10 text-slate-500 cursor-not-allowed'
                : 'bg-gradient-to-r from-cyan-400 to-emerald-400 text-slate-950 hover:brightness-105 shadow-lg shadow-cyan-500/20'
            }`}
          >
            {isValidating ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Testando...</span>
              </>
            ) : (
              <>
                <Check className="w-4 h-4" />
                <span>Salvar e Validar</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Free Alternative Providers (Groq + OpenRouter) */}
      <div className="fd-panel rounded-3xl p-6 sm:p-7 space-y-5">
        <div className="flex items-start justify-between border-b border-white/10 pb-3.5 gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-500/15 border border-amber-500/25 flex items-center justify-center shrink-0">
              <Zap className="w-4 h-4 text-amber-300" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">
                Provedores Alternativos Gratuitos (Zero Erro 503 do Google)
              </h3>
              <p className="text-xs text-slate-400">
                Se o Gemini estiver congestionado, o app analisa os quadros do vídeo usando estes servidores.
              </p>
            </div>
          </div>
        </div>

        {/* Groq Card */}
        <div className="fd-card rounded-2xl p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-300" />
              <span className="text-xs font-bold text-white">1. Groq Cloud (Llama 4 Vision — Grátis e Ultra-Rápido)</span>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <a
                href="https://console.groq.com/keys"
                target="_blank"
                rel="noreferrer"
                className="text-cyan-300 hover:text-cyan-200 underline inline-flex items-center gap-1"
              >
                <span>Criar chave grátis</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              {groqStatus === 'valid' && (
                <span className="text-emerald-300 font-mono font-semibold">Conectada</span>
              )}
              {groqStatus === 'invalid' && (
                <span className="text-rose-300 font-mono font-semibold">Inválida</span>
              )}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative flex-1">
              <input
                type={showGroqKey ? 'text' : 'password'}
                value={groqKey}
                onChange={(e) => {
                  setGroqKey(e.target.value);
                  setGroqStatus('idle');
                }}
                placeholder="gsk_... (cole sua chave gratuita do Groq)"
                className="w-full bg-black/50 border border-white/15 rounded-xl pl-3.5 pr-10 py-2.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400 font-mono"
              />
              <button
                type="button"
                onClick={() => setShowGroqKey(!showGroqKey)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-white"
              >
                {showGroqKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>
            <button
              type="button"
              disabled={isValidatingGroq}
              onClick={handleSaveGroq}
              className="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-semibold flex items-center justify-center gap-1.5 shrink-0"
            >
              {isValidatingGroq ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Check className="w-3.5 h-3.5 text-emerald-300" />
              )}
              <span>{groqKey.trim() ? 'Validar Groq' : savedGroqKey ? 'Remover' : 'Salvar'}</span>
            </button>
          </div>
        </div>

        {/* OpenRouter Card */}
        <div className="fd-card rounded-2xl p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-300" />
              <span className="text-xs font-bold text-white">2. OpenRouter (Gemma 3 / Qwen VL :free — Grátis)</span>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <a
                href="https://openrouter.ai/keys"
                target="_blank"
                rel="noreferrer"
                className="text-cyan-300 hover:text-cyan-200 underline inline-flex items-center gap-1"
              >
                <span>Criar chave grátis</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              {openRouterStatus === 'valid' && (
                <span className="text-emerald-300 font-mono font-semibold">Conectada</span>
              )}
              {openRouterStatus === 'invalid' && (
                <span className="text-rose-300 font-mono font-semibold">Inválida</span>
              )}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-2">
            <div className="relative flex-1">
              <input
                type={showOpenRouterKey ? 'text' : 'password'}
                value={openRouterKey}
                onChange={(e) => {
                  setOpenRouterKey(e.target.value);
                  setOpenRouterStatus('idle');
                }}
                placeholder="sk-or-... (cole sua chave gratuita do OpenRouter)"
                className="w-full bg-black/50 border border-white/15 rounded-xl pl-3.5 pr-10 py-2.5 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyan-400 font-mono"
              />
              <button
                type="button"
                onClick={() => setShowOpenRouterKey(!showOpenRouterKey)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-white"
              >
                {showOpenRouterKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>
            <button
              type="button"
              disabled={isValidatingOpenRouter}
              onClick={handleSaveOpenRouter}
              className="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-white text-xs font-semibold flex items-center justify-center gap-1.5 shrink-0"
            >
              {isValidatingOpenRouter ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Check className="w-3.5 h-3.5 text-emerald-300" />
              )}
              <span>
                {openRouterKey.trim()
                  ? 'Validar OpenRouter'
                  : savedOpenRouterKey
                  ? 'Remover'
                  : 'Salvar'}
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* Tutorial */}
      <div className="fd-panel rounded-3xl p-6 sm:p-7 space-y-4 text-xs">
        <div className="flex items-center gap-2 text-white font-semibold text-sm">
          <Sparkles className="w-4 h-4 text-cyan-300" />
          <span>Como obter sua chave gratuita do Google Gemini</span>
        </div>

        <div className="space-y-2.5">
          <div className="fd-card rounded-2xl p-4 flex items-start gap-3">
            <span className="font-mono font-bold text-cyan-300 shrink-0">01.</span>
            <div className="space-y-1 flex-1">
              <span className="text-white font-semibold block">Acesse o Google AI Studio</span>
              <a
                href="https://aistudio.google.com/app/apikey"
                target="_blank"
                rel="noreferrer"
                className="text-cyan-300 hover:text-cyan-200 underline font-mono inline-flex items-center gap-1 break-all"
              >
                https://aistudio.google.com/app/apikey
                <ExternalLink className="w-3 h-3 shrink-0" />
              </a>
            </div>
          </div>

          <div className="fd-card rounded-2xl p-4 flex items-start gap-3">
            <span className="font-mono font-bold text-cyan-300 shrink-0">02.</span>
            <div className="space-y-1">
              <span className="text-white font-semibold block">Entre com sua conta Google</span>
              <p className="text-slate-300 leading-relaxed">
                O acesso gratuito permite que cada membro da equipe crie sua própria chave para escanear os rounds.
              </p>
            </div>
          </div>

          <div className="fd-card rounded-2xl p-4 flex items-start gap-3">
            <span className="font-mono font-bold text-cyan-300 shrink-0">03.</span>
            <div className="space-y-1">
              <span className="text-white font-semibold block">Clique em "Create API key"</span>
              <p className="text-slate-300 leading-relaxed">
                Copie a chave gerada e cole no campo acima para liberar as análises imediatamente.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
