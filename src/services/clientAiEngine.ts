import { GoogleGenAI, MediaResolution, ThinkingLevel, Type } from '@google/genai';
import type { FolderMissionReference, ReferencePhotoItem } from '../components/FolderReferenceView';
import { BIOGLOW_OFFICIAL_MISSIONS } from '../data/bioglowMissions';
import { parseFolderRelativePath } from '../utils/folderParser';
import type { AiTrainingExample } from '../types/fll';

let backendAvailablePromise: Promise<boolean> | null = null;

/**
 * Checks whether the Express backend (/api/*) is active.
 * On GitHub Pages or static hosting, returns false quickly and caches the result.
 */
export async function hasBackendServer(forceRefresh = false): Promise<boolean> {
  if (typeof window === 'undefined') return false;
  if (window.location.protocol === 'file:') return false;

  if (!backendAvailablePromise || forceRefresh) {
    backendAvailablePromise = (async () => {
      try {
        const ctrl = new AbortController();
        const timer = window.setTimeout(() => ctrl.abort(), 1800);
        const res = await fetch('./api/health', {
          method: 'GET',
          cache: 'no-store',
          signal: ctrl.signal,
        });
        window.clearTimeout(timer);

        const ctype = (res.headers.get('content-type') || '').toLowerCase();
        if (!res.ok || !ctype.includes('application/json')) {
          return false;
        }
        const data = await res.json();
        return data?.status === 'ok';
      } catch {
        return false;
      }
    })();
  }

  return backendAvailablePromise;
}

export const GEMINI_API_KEY_STORAGE_KEY = 'francodroid_user_gemini_api_key';
export const GROQ_API_KEY_STORAGE_KEY = 'francodroid_user_groq_api_key';
export const OPENROUTER_API_KEY_STORAGE_KEY = 'francodroid_user_openrouter_api_key';
export const AI_PROVIDER_PREF_KEY = 'francodroid_ai_provider_pref';
export const PREFERRED_GEMINI_MODEL_KEY = 'francodroid_preferred_gemini_model';

export type AiProviderPref = 'auto' | 'gemini' | 'groq' | 'openrouter';

export const SELECTABLE_GEMINI_MODELS = [
  { id: 'auto', label: 'Automático (Alterna modelos se der 503)' },
  { id: 'gemini-3.1-flash-lite', label: 'Gemini 3.1 Flash-Lite (Mais leve / Menos 503)' },
  { id: 'gemini-3.8-flash', label: 'Gemini 3.8 Flash (Rápido)' },
  { id: 'gemini-4-hydrogen', label: 'Gemini 4 Hydrogen (Padrão)' },
  { id: 'gemma-3-27b-it', label: 'Gemma 3 27B (Servidor alternativo Google)' },
  { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash (Legado se liberado)' },
];

export function hasAnyConfiguredAiProvider(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    return Boolean(
      (localStorage.getItem(GEMINI_API_KEY_STORAGE_KEY) || '').trim() ||
        (localStorage.getItem(GROQ_API_KEY_STORAGE_KEY) || '').trim() ||
        (localStorage.getItem(OPENROUTER_API_KEY_STORAGE_KEY) || '').trim()
    );
  } catch {
    return false;
  }
}

export function getSavedProviderConfig(): {
  geminiKey: string;
  groqKey: string;
  openRouterKey: string;
  providerPref: AiProviderPref;
  preferredGeminiModel: string;
} {
  if (typeof window === 'undefined') {
    return {
      geminiKey: '',
      groqKey: '',
      openRouterKey: '',
      providerPref: 'auto',
      preferredGeminiModel: 'auto',
    };
  }
  try {
    const prefRaw = localStorage.getItem(AI_PROVIDER_PREF_KEY) || 'auto';
    const providerPref: AiProviderPref =
      prefRaw === 'gemini' || prefRaw === 'groq' || prefRaw === 'openrouter' ? prefRaw : 'auto';
    return {
      geminiKey: (localStorage.getItem(GEMINI_API_KEY_STORAGE_KEY) || '').trim(),
      groqKey: (localStorage.getItem(GROQ_API_KEY_STORAGE_KEY) || '').trim(),
      openRouterKey: (localStorage.getItem(OPENROUTER_API_KEY_STORAGE_KEY) || '').trim(),
      providerPref,
      preferredGeminiModel: (localStorage.getItem(PREFERRED_GEMINI_MODEL_KEY) || 'auto').trim() || 'auto',
    };
  } catch {
    return {
      geminiKey: '',
      groqKey: '',
      openRouterKey: '',
      providerPref: 'auto',
      preferredGeminiModel: 'auto',
    };
  }
}

const EXCLUDED_MODEL_KEYWORDS = [
  'embedding',
  'tts',
  'transcribe',
  'live',
  'image',
  'banana',
  'lyria',
  'veo',
  'omni',
  'decision',
  'deep-research',
  'robotics',
  'computer-use',
  'learnlm',
  'aqa',
  'bison',
  'gecko',
  'gemini-1.0',
  'gemini-1.5',
  'gemini-2.0',
  'thinking',
  '-pro',
  'carbon',
  'neon',
  'argon',
];

const PREFERRED_MODEL_ORDER = [
  'gemini-4-hydrogen',
  'gemini-3.8-flash',
  'gemini-3.1-flash-lite',
  'gemini-3-flash',
  'gemini-3-flash-preview',
  'gemini-2.5-flash',
  'gemini-2.5-flash-lite',
  'gemma-3-27b-it',
  'gemma-3-12b-it',
];

const DEFAULT_FALLBACK_MODELS = [
  'gemini-4-hydrogen',
  'gemini-3.8-flash',
  'gemini-3.1-flash-lite',
  'gemini-3-flash',
  'gemma-3-27b-it',
];

function scoreModelRank(name: string): number {
  const idx = PREFERRED_MODEL_ORDER.indexOf(name);
  if (idx !== -1) return idx;
  if (name.includes('flash-lite')) return 15;
  if (name.includes('flash')) return 20;
  if (name.startsWith('gemma-3-27b')) return 35;
  if (name.startsWith('gemma-3-12b')) return 38;
  if (name.startsWith('gemma-')) return 45;
  return 60;
}

function extractJsonObjectFromText(rawText: string): any {
  const cleaned = String(rawText || '')
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  try {
    return JSON.parse(cleaned || '{}');
  } catch {
    const match = cleaned.match(/\{[\s\S]*\}/);
    if (match) {
      return JSON.parse(match[0]);
    }
    throw new Error('A IA retornou uma resposta fora do formato JSON esperado.');
  }
}

function extractGeminiErrorText(err: any): string {
  const raw = typeof err === 'string' ? err : String(err?.message || err || '');
  const trimmed = raw.trim();
  if (trimmed.startsWith('{') || trimmed.includes('"error"')) {
    try {
      const jsonMatch = trimmed.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        if (parsed?.error?.message) {
          const code = parsed.error.code ? `[${parsed.error.code}] ` : '';
          const status = parsed.error.status ? ` (${parsed.error.status})` : '';
          return `${code}${parsed.error.message}${status}`;
        }
      }
    } catch {
      // ignore JSON parse issues
    }
  }
  return raw;
}

export function isTransientOverloadError(err: any): boolean {
  const text = extractGeminiErrorText(err).toLowerCase();
  const status = Number(err?.status || err?.code || 0);
  return (
    status === 503 ||
    status === 429 ||
    status === 502 ||
    status === 504 ||
    text.includes('503') ||
    text.includes('unavailable') ||
    text.includes('high demand') ||
    text.includes('overloaded') ||
    text.includes('try again later') ||
    text.includes('temporarily') ||
    text.includes('resource_exhausted') ||
    text.includes('429') ||
    text.includes('tempo limite')
  );
}

export function formatClientGeminiError(err: any): string {
  const extracted = extractGeminiErrorText(err);
  const lower = extracted.toLowerCase();

  if (
    lower.includes('api_key_invalid') ||
    lower.includes('permission_denied') ||
    lower.includes('[401]') ||
    lower.includes('[403]')
  ) {
    return 'Sua chave da API Gemini parece inválida ou sem permissão. Verifique-a na aba "Configurar IA".';
  }
  if (
    lower.includes('503') ||
    lower.includes('unavailable') ||
    lower.includes('high demand') ||
    lower.includes('overloaded')
  ) {
    return 'Os servidores do Gemini estão com pico temporário de acesso (erro 503 — alta demanda). Tentamos automaticamente os modelos alternativos, mas todos estão ocupados agora. Aguarde ~15–30 segundos e clique em Analisar novamente.';
  }
  if (lower.includes('resource_exhausted') || lower.includes('429') || lower.includes('quota')) {
    return 'Cota gratuita da API Gemini temporariamente atingida. Aguarde ~20 segundos ou use "Dividir em Partes" / corte um trecho menor.';
  }
  if (
    lower.includes('payload size exceeds') ||
    lower.includes('token count') ||
    lower.includes('too large') ||
    lower.includes('413')
  ) {
    return 'Os dados enviados (vídeo + fotos do gabarito) excederam o limite da API gratuita. Tente informar as missões feitas ou cortar um trecho menor.';
  }
  if (lower.includes('404') || lower.includes('not_found')) {
    return 'O modelo selecionado não está disponível na sua conta Gemini no momento. Tente novamente em instantes.';
  }
  if (lower.includes('tempo limite')) {
    return extracted;
  }
  return extracted || 'Falha ao processar análise com a IA Gemini.';
}

async function discoverFlashModelsDirect(apiKey: string): Promise<string[]> {
  try {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), 6000);
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?pageSize=100&key=${encodeURIComponent(apiKey.trim())}`,
      { signal: ctrl.signal }
    );
    window.clearTimeout(timer);

    if (!res.ok) {
      return [...DEFAULT_FALLBACK_MODELS];
    }

    const data = await res.json();
    const models: any[] = Array.isArray(data?.models) ? data.models : [];

    const valid = models
      .filter((m) => {
        const methods: string[] = Array.isArray(m?.supportedGenerationMethods)
          ? m.supportedGenerationMethods
          : [];
        if (!methods.includes('generateContent')) return false;
        const shortName = String(m?.name || '')
          .replace(/^models\//, '')
          .toLowerCase();
        if (!shortName.startsWith('gemini-') && !shortName.startsWith('gemma-3-')) return false;
        if (shortName.startsWith('gemma-3-') && !shortName.includes('27b') && !shortName.includes('12b')) {
          return false;
        }
        return !EXCLUDED_MODEL_KEYWORDS.some((kw) => shortName.includes(kw));
      })
      .map((m) => String(m.name).replace(/^models\//, ''));

    const ranked = Array.from(new Set(valid)).sort(
      (a, b) => scoreModelRank(a) - scoreModelRank(b)
    );
    return ranked.length > 0 ? ranked.slice(0, 6) : [...DEFAULT_FALLBACK_MODELS];
  } catch {
    return [...DEFAULT_FALLBACK_MODELS];
  }
}

export async function validateGroqKeyDirect(
  rawKey: string
): Promise<{ valid: boolean; message: string }> {
  const key = rawKey.trim();
  if (!key || key.length < 16 || /\s/.test(key)) {
    return { valid: false, message: 'Chave Groq inválida ou incompleta (começa com gsk_).' };
  }
  try {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), 6500);
    const res = await fetch('https://api.groq.com/openai/v1/models', {
      headers: { Authorization: `Bearer ${key}` },
      signal: ctrl.signal,
    });
    window.clearTimeout(timer);
    if (res.ok) {
      return { valid: true, message: 'Chave Groq conectada! Alternativa anti-503 pronta.' };
    }
    return { valid: false, message: `Chave Groq recusada (HTTP ${res.status}). Confira em console.groq.com.` };
  } catch {
    return { valid: false, message: 'Falha de conexão com a API Groq.' };
  }
}

export async function validateOpenRouterKeyDirect(
  rawKey: string
): Promise<{ valid: boolean; message: string }> {
  const key = rawKey.trim();
  if (!key || key.length < 16 || /\s/.test(key)) {
    return { valid: false, message: 'Chave OpenRouter inválida ou incompleta (começa com sk-or-).' };
  }
  try {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), 6500);
    const res = await fetch('https://openrouter.ai/api/v1/auth/key', {
      headers: { Authorization: `Bearer ${key}` },
      signal: ctrl.signal,
    });
    window.clearTimeout(timer);
    if (res.ok) {
      return { valid: true, message: 'Chave OpenRouter conectada! Modelos gratuitos prontos.' };
    }
    return { valid: false, message: `Chave OpenRouter recusada (HTTP ${res.status}). Confira em openrouter.ai.` };
  } catch {
    return { valid: false, message: 'Falha de conexão com a API OpenRouter.' };
  }
}

export async function validateGeminiKeyDirect(
  rawKey: string
): Promise<{ valid: boolean; message: string; quotaWarning?: boolean }> {
  const apiKey = rawKey.trim();
  if (!apiKey) {
    return { valid: false, message: 'Chave da API não fornecida.' };
  }
  if (apiKey.length < 16 || /\s/.test(apiKey)) {
    return {
      valid: false,
      message: 'Chave incompleta ou com espaços. Copie a chave inteira gerada no Google AI Studio.',
    };
  }

  try {
    const ctrl = new AbortController();
    const timer = window.setTimeout(() => ctrl.abort(), 6500);
    const resp = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models?pageSize=20&key=${encodeURIComponent(apiKey)}`,
      { signal: ctrl.signal }
    );
    window.clearTimeout(timer);

    if (resp.ok) {
      return {
        valid: true,
        message: 'Chave Gemini validada e conectada com sucesso!',
      };
    }

    const errJson = await resp.json().catch(() => ({}));
    const rawMsg = String(errJson?.error?.message || `HTTP ${resp.status}`);
    const lower = rawMsg.toLowerCase();

    if (resp.status === 429 || lower.includes('quota') || lower.includes('resource_exhausted')) {
      return {
        valid: true,
        quotaWarning: true,
        message:
          'Chave conectada! (A cota gratuita está momentaneamente cheia, aguarde ~20s se ocorrer erro ao analisar).',
      };
    }

    if (resp.status === 503 || lower.includes('unavailable') || lower.includes('high demand')) {
      return {
        valid: true,
        quotaWarning: true,
        message:
          'Chave reconhecida! (Servidor do Google com pico temporário de acesso 503 — tente analisar em alguns segundos).',
      };
    }

    if (resp.status === 400 || resp.status === 401 || resp.status === 403 || lower.includes('api_key_invalid')) {
      return {
        valid: false,
        message: 'Chave inválida ou bloqueada. Copie a chave correta no Google AI Studio.',
      };
    }

    return { valid: false, message: `Erro na chave Gemini: ${rawMsg}` };
  } catch {
    return {
      valid: false,
      message: 'Não foi possível alcançar a API do Google Gemini. Confira sua internet.',
    };
  }
}

export async function loadStaticTeamGabarito(): Promise<{
  missionCount: number;
  photoCount: number;
  references: Record<string, FolderMissionReference>;
}> {
  const res = await fetch('./team-gabarito.json', { cache: 'no-store' });
  if (!res.ok) {
    throw new Error('Arquivo estático do gabarito da equipe não encontrado.');
  }
  const raw = await res.json();
  const rawRefs: Record<string, any> = raw?.references || {};
  const hydrated: Record<string, FolderMissionReference> = {};
  let totalPhotos = 0;

  for (const code of Object.keys(rawRefs)) {
    const src = rawRefs[code];
    const ref: FolderMissionReference = {
      missionCode: code,
      completaFiles: [],
      parcialFiles: [],
      incompletaFiles: [],
      pointTiers: {},
    };

    const tiers = src?.pointTiers || {};
    for (const tierKey of Object.keys(tiers)) {
      const t = tiers[tierKey];
      const files: ReferencePhotoItem[] = (Array.isArray(t?.files) ? t.files : []).map((f: any) => ({
        name: String(f.name || 'foto.jpg'),
        url: String(f.base64 || f.url || ''),
        base64: f.base64 ? String(f.base64) : undefined,
        points: typeof f.points === 'number' ? f.points : Number(t?.points) || 0,
        tierLabel: String(f.tierLabel || t?.label || tierKey),
      }));

      ref.pointTiers![tierKey] = {
        label: String(t?.label || tierKey),
        points: Number(t?.points) || 0,
        files,
      };

      const parsed = parseFolderRelativePath(`Drive/${code}/${t?.label || tierKey}/foto.jpg`);
      const state = parsed.isValid
        ? parsed.state
        : Number(t?.points) === 0
        ? 'incompleta'
        : 'completa';

      if (state === 'completa') ref.completaFiles.push(...files);
      else if (state === 'parcial') ref.parcialFiles.push(...files);
      else ref.incompletaFiles.push(...files);

      totalPhotos += files.length;
    }

    hydrated[code] = ref;
  }

  return {
    missionCount: Object.keys(hydrated).length,
    photoCount: totalPhotos,
    references: hydrated,
  };
}

function formatSecShort(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${String(m).padStart(2, '0')}:${String(rem).padStart(2, '0')}`;
}

function fileToBase64Raw(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || '');
      const idx = result.indexOf(';base64,');
      resolve(idx >= 0 ? result.slice(idx + 8) : result);
    };
    reader.onerror = () => reject(new Error('Falha ao ler arquivo no navegador.'));
    reader.readAsDataURL(file);
  });
}

function probeBrowserVideoDuration(file: File): Promise<number> {
  return new Promise((resolve) => {
    const video = document.createElement('video');
    video.preload = 'metadata';
    const url = URL.createObjectURL(file);
    const done = (dur: number) => {
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(dur) && dur > 0 ? dur : 0);
    };
    video.onloadedmetadata = () => done(video.duration);
    video.onerror = () => done(0);
    window.setTimeout(() => done(0), 5000);
    video.src = url;
  });
}

/**
 * Extracts crisp 720p JPEG keyframes with exact timestamps from a local File inside the browser.
 * Allows analyzing even large 60-100 MB videos on static GitHub Pages without FFmpeg!
 */
async function extractVideoSegmentKeyframes(
  file: File,
  startSec: number,
  endSec: number,
  maxFrames = 10,
  signal?: AbortSignal
): Promise<Array<{ timestampSec: number; base64Data: string }>> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';

    const objectUrl = URL.createObjectURL(file);
    const frames: Array<{ timestampSec: number; base64Data: string }> = [];

    const cleanup = () => {
      URL.revokeObjectURL(objectUrl);
      video.removeAttribute('src');
      video.load();
    };

    if (signal?.aborted) {
      cleanup();
      reject(new Error('ABORTED'));
      return;
    }

    const abortHandler = () => {
      cleanup();
      reject(new Error('ABORTED'));
    };
    signal?.addEventListener('abort', abortHandler, { once: true });

    video.onerror = () => {
      signal?.removeEventListener('abort', abortHandler);
      cleanup();
      reject(new Error('O navegador não conseguiu decodificar este vídeo. Tente converter para MP4 padrão.'));
    };

    video.onloadeddata = async () => {
      try {
        const safeDur = Math.max(1, video.duration || endSec || 30);
        const clampedStart = Math.max(0, Math.min(startSec, safeDur - 0.5));
        const clampedEnd = Math.max(clampedStart + 0.5, Math.min(endSec, safeDur));
        const span = clampedEnd - clampedStart;

        const targetCount = Math.max(3, Math.min(maxFrames, Math.ceil(span / 2.2) + 1));
        const step = targetCount > 1 ? span / (targetCount - 1) : 0;

        const srcW = video.videoWidth || 640;
        const srcH = video.videoHeight || 360;
        const targetW = Math.min(720, srcW);
        const targetH = Math.max(2, Math.round((srcH * targetW) / srcW));

        const canvas = document.createElement('canvas');
        canvas.width = targetW;
        canvas.height = targetH;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Canvas indisponível para extrair quadros.');

        for (let i = 0; i < targetCount; i++) {
          if (signal?.aborted) throw new Error('ABORTED');
          const t = Math.min(safeDur - 0.15, clampedStart + i * step);

          await new Promise<void>((resSeek) => {
            const onSeeked = () => {
              video.removeEventListener('seeked', onSeeked);
              resSeek();
            };
            video.addEventListener('seeked', onSeeked, { once: true });
            video.currentTime = t;
            window.setTimeout(resSeek, 900);
          });

          ctx.drawImage(video, 0, 0, targetW, targetH);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.76);
          const b64 = dataUrl.split(';base64,')[1] || '';
          if (b64) {
            frames.push({ timestampSec: t, base64Data: b64 });
          }
        }

        signal?.removeEventListener('abort', abortHandler);
        cleanup();
        resolve(frames);
      } catch (err) {
        signal?.removeEventListener('abort', abortHandler);
        cleanup();
        reject(err);
      }
    };

    video.src = objectUrl;
  });
}

function sanitizeClientAnalysisOutput(
  rawParsed: any,
  allowedMissions: Array<{ code: string; maxScore: number; validScores?: number[] }>,
  hasGabarito: boolean,
  trainingRulesApplied: number,
  informedCodes: string[]
) {
  const missionMap = new Map(allowedMissions.map((m) => [m.code.toUpperCase(), m]));
  const rawList = Array.isArray(rawParsed?.missionsDetected) ? rawParsed.missionsDetected : [];
  const deduped = new Map<string, any>();

  for (const item of rawList) {
    const code = String(item?.missionCode || '').trim().toUpperCase();
    if (!missionMap.has(code)) continue;

    const rawStatus = String(item?.status || '').toLowerCase();
    const rawScore = Number(item?.estimatedScore) || 0;
    if (rawStatus === 'not_attempted' && informedCodes.length === 0) continue;

    const meta = missionMap.get(code)!;
    let clampedScore = Math.max(0, Math.min(meta.maxScore || 40, Math.round(rawScore)));
    if (
      Array.isArray(meta.validScores) &&
      meta.validScores.length > 0 &&
      !meta.validScores.includes(clampedScore)
    ) {
      clampedScore = meta.validScores.reduce((prev, curr) =>
        Math.abs(curr - clampedScore) < Math.abs(prev - clampedScore) ? curr : prev
      );
    }

    const confidence = Math.max(
      0,
      Math.min(100, Math.round(Number(item?.confidence) || (hasGabarito ? 75 : 55)))
    );

    let status: 'completed' | 'partial' | 'failed' | 'not_attempted' = 'failed';
    if (clampedScore <= 0) {
      status = rawStatus === 'not_attempted' ? 'not_attempted' : 'failed';
    } else if (meta.maxScore > 0 && clampedScore >= meta.maxScore) {
      status = 'completed';
    } else {
      status = 'partial';
    }

    const candidate = {
      missionCode: code,
      status,
      estimatedScore: clampedScore,
      timestampEstimate: item?.timestampEstimate ? String(item.timestampEstimate) : undefined,
      evidenceDescription:
        String(item?.evidenceDescription || '').trim() || 'Estado identificado na análise visual.',
      confidence,
    };

    const prev = deduped.get(code);
    if (!prev || candidate.confidence > prev.confidence) {
      deduped.set(code, candidate);
    }
  }

  const sanitizedMissions = Array.from(deduped.values()).sort((a, b) =>
    a.missionCode.localeCompare(b.missionCode)
  );
  const computedTotalScore = sanitizedMissions.reduce(
    (sum, m) => sum + (Number(m.estimatedScore) || 0),
    0
  );
  const totalTimeSeconds = Math.max(
    0,
    Math.min(300, Math.round(Number(rawParsed?.totalTimeSeconds) || 0))
  );

  return {
    estimatedTotalScore: computedTotalScore,
    totalTimeFormatted: String(rawParsed?.totalTimeFormatted || '00:00'),
    totalTimeSeconds,
    overallConfidence: Math.max(
      0,
      Math.min(100, Math.round(Number(rawParsed?.overallConfidence) || (hasGabarito ? 78 : 58)))
    ),
    summary: String(rawParsed?.summary || 'Análise concluída.'),
    trajectoryNotes: String(rawParsed?.trajectoryNotes || ''),
    mechanismNotes: String(rawParsed?.mechanismNotes || ''),
    missionsDetected: sanitizedMissions,
    usedGabarito: hasGabarito,
    trainingRulesApplied,
    gabaritoWarning: hasGabarito
      ? undefined
      : 'Análise feita sem fotos de gabarito (apenas com o Livro de Regras BIOGLOW). Confira e corrija qualquer erro antes de salvar.',
  };
}

function mergeClientPartAnalyses(
  partResults: Array<{ startSec: number; endSec: number; sanitized: any }>,
  totalDurationSec: number,
  hasGabarito: boolean,
  trainingRulesApplied: number
) {
  const mergedMap = new Map<string, any>();

  partResults.forEach(({ startSec, endSec, sanitized }, idx) => {
    const tag = `[Parte ${idx + 1}: ${formatSecShort(startSec)}–${formatSecShort(endSec)}]`;
    const list = Array.isArray(sanitized?.missionsDetected) ? sanitized.missionsDetected : [];

    for (const m of list) {
      const code = String(m.missionCode || '').toUpperCase();
      if (!code) continue;

      const candidate = {
        ...m,
        timestampEstimate: m.timestampEstimate ? `${tag} ${m.timestampEstimate}` : tag,
      };

      const prev = mergedMap.get(code);
      if (!prev) {
        mergedMap.set(code, candidate);
      } else {
        const prevScore = Number(prev.estimatedScore) || 0;
        const currScore = Number(candidate.estimatedScore) || 0;
        if (
          currScore > prevScore ||
          (currScore === prevScore && Number(candidate.confidence || 0) >= Number(prev.confidence || 0))
        ) {
          const combinedEvidence =
            prev.evidenceDescription && candidate.evidenceDescription !== prev.evidenceDescription
              ? `${candidate.evidenceDescription} (${prev.timestampEstimate}: ${prev.evidenceDescription})`
              : candidate.evidenceDescription;
          mergedMap.set(code, {
            ...candidate,
            evidenceDescription: combinedEvidence.slice(0, 450),
          });
        }
      }
    }
  });

  const mergedMissions = Array.from(mergedMap.values()).sort((a, b) =>
    a.missionCode.localeCompare(b.missionCode)
  );
  const totalScore = mergedMissions.reduce((s, m) => s + (Number(m.estimatedScore) || 0), 0);
  const avgConf =
    partResults.length > 0
      ? Math.round(
          partResults.reduce((s, p) => s + (Number(p.sanitized?.overallConfidence) || 70), 0) /
            partResults.length
        )
      : 70;
  const finalSec = totalDurationSec > 0 ? Math.round(totalDurationSec) : 0;

  return {
    estimatedTotalScore: totalScore,
    totalTimeFormatted: formatSecShort(finalSec),
    totalTimeSeconds: finalSec,
    overallConfidence: avgConf,
    summary: `Análise consolidada em ${partResults.length} partes (modo navegador) — ${partResults
      .map((p, i) => `Parte ${i + 1}: ${p.sanitized?.summary || 'analisada'}`)
      .join(' | ')}`.slice(0, 1200),
    trajectoryNotes: partResults
      .map((p) => p.sanitized?.trajectoryNotes)
      .filter(Boolean)
      .join(' ')
      .slice(0, 600),
    mechanismNotes: partResults
      .map((p) => p.sanitized?.mechanismNotes)
      .filter(Boolean)
      .join(' ')
      .slice(0, 600),
    missionsDetected: mergedMissions,
    usedGabarito: hasGabarito,
    trainingRulesApplied,
    partsAnalyzed: partResults.length,
    gabaritoWarning: hasGabarito
      ? undefined
      : 'Análise feita sem fotos de gabarito (apenas com o Livro de Regras BIOGLOW). Confira e corrija qualquer erro antes de salvar.',
  };
}

export async function analyzeMediaClientSide(params: {
  mediaFile: File;
  isVideo: boolean;
  videoDuration: number;
  informedMissions: string[];
  references: Record<string, FolderMissionReference>;
  trainingExamples: AiTrainingExample[];
  splitMode: 'auto' | 'parts' | 'single';
  apiKey: string;
  groqApiKey?: string;
  openRouterApiKey?: string;
  providerPref?: AiProviderPref;
  preferredGeminiModel?: string;
  forceKeyframes?: boolean;
  signal?: AbortSignal;
  onProgress: (percent: number, message: string) => void;
}) {
  const {
    mediaFile,
    isVideo,
    informedMissions,
    references,
    trainingExamples,
    splitMode,
    apiKey,
    signal,
    onProgress,
  } = params;

  const cleanedInformed = informedMissions.map((c) => c.trim().toUpperCase()).filter(Boolean);
  const activeMissions =
    cleanedInformed.length > 0
      ? BIOGLOW_OFFICIAL_MISSIONS.filter((m) => cleanedInformed.includes(m.code.toUpperCase()))
      : BIOGLOW_OFFICIAL_MISSIONS;
  const allowedCodesList = activeMissions.map((m) => m.code.toUpperCase());

  // Build concise gabarito reference parts
  const refImages: Array<{ missionCode: string; tierLabel: string; points: number; base64: string }> = [];
  for (const code of Object.keys(references).sort()) {
    if (!allowedCodesList.includes(code)) continue;
    const item = references[code];
    if (item?.pointTiers) {
      for (const tier of Object.values(item.pointTiers)) {
        const sample = tier.files.find((f) => Boolean(f.base64));
        if (sample?.base64) {
          refImages.push({
            missionCode: code,
            tierLabel: tier.label,
            points: tier.points,
            base64: sample.base64,
          });
        }
      }
    }
  }

  const refCap = cleanedInformed.length > 0 ? 8 : 5;
  const relevantRefImages = refImages.slice(0, refCap);
  const hasGabarito = relevantRefImages.length > 0;

  const referenceParts: any[] = [];
  if (hasGabarito) {
    referenceParts.push({ text: 'FOTOS DO GABARITO DA EQUIPE (compare o estado final):' });
    for (const refImg of relevantRefImages) {
      let refMime = 'image/jpeg';
      let refData = refImg.base64;
      if (refData.includes(';base64,')) {
        const sp = refData.split(';base64,');
        refMime = sp[0].replace('data:', '') || 'image/jpeg';
        refData = sp[1];
      }
      referenceParts.push({
        text: `[GABARITO] ${refImg.missionCode} | Pasta: "${refImg.tierLabel}" | Pontos: ${refImg.points}`,
      });
      referenceParts.push({ inlineData: { mimeType: refMime, data: refData } });
    }
  }

  // Training feedback rules
  const relevantTraining = trainingExamples
    .filter((t) => t.missionCode === 'GERAL' || allowedCodesList.includes(t.missionCode.toUpperCase()))
    .slice(0, 12);
  const trainingParts: any[] = [];
  let trainingPromptSection = '';

  if (relevantTraining.length > 0) {
    const lines = relevantTraining.map(
      (t, i) =>
        `${i + 1}. [${t.missionCode}] Erro: "${t.aiPredictedSummary}" -> Correto: "${t.expectedSummary}"${
          t.userExplanation ? ` | Regra: ${t.userExplanation}` : ''
        }`
    );
    trainingPromptSection = `\n*** APRENDIZADO DA EQUIPE (OBRIGATÓRIO) ***:\n${lines.join('\n')}\n`;
    const snapEx = relevantTraining.find((t) => Boolean(t.snapshotBase64));
    if (snapEx?.snapshotBase64) {
      const sp = snapEx.snapshotBase64.split(';base64,');
      const snapMime = sp[0].replace('data:', '') || 'image/jpeg';
      const snapData = sp[1] || snapEx.snapshotBase64;
      trainingParts.push({
        text: `[EXEMPLO DE CORREÇÃO] ${snapEx.missionCode}: ${snapEx.expectedSummary}. ${snapEx.userExplanation}`,
      });
      trainingParts.push({ inlineData: { mimeType: snapMime, data: snapData } });
    }
  }

  const scopeInstruction =
    cleanedInformed.length > 0
      ? `O OPERADOR INFORMOU APENAS: [${cleanedInformed.join(', ')}]. Avalie EXCLUSIVAMENTE estas missões.`
      : `MODO AUTOMÁTICO ANTI-ALUCINAÇÃO:
1. Todos os 15 modelos LEGO ficam no tapete. Modelo parado ao fundo NÃO conta!
2. SÓ inclua em "missionsDetected" se o robô REALMENTE tocou/alterou aquele modelo no vídeo.
3. NUNCA inclua M10 automaticamente.`;

  const gabaritoNote = hasGabarito
    ? `GABARITO ATIVO (${relevantRefImages.length} fotos anexadas): compare estado final com as fotos.`
    : `SEM GABARITO: avalie pelo guia abaixo sem dar bônus sem evidência visível.`;

  const compactMissionsGuide = activeMissions
    .map(
      (m) =>
        `${m.code} (${m.name}, válidos:[${(m.validScores || [0, m.maxScore]).join(',')}]): ${
          m.visualGuide || m.description || ''
        }`
    )
    .join('\n');

  const buildPrompt = (segmentNote = '') => `
Árbitro FLL (Temporada BIOGLOW).
${segmentNote ? `TRECHO: ${segmentNote}\n` : ''}${scopeInstruction}
${gabaritoNote}
${trainingPromptSection}
REGRAS DAS MISSÕES:
${compactMissionsGuide}

DETALHES CRÍTICOS:
- M01: Drone fora do tapete = 20 pts; + peça vermelha LiDAR dentro da caixinha = 30 pts.
- M02: Haste verde curva centro-esquerda (NÃO confundir com M08 trepadeira). 1=10, 2=20, 3=30 pts.
`;

  const savedCfg = getSavedProviderConfig();
  const effectiveGeminiKey = (apiKey || savedCfg.geminiKey || '').trim();
  const effectiveGroqKey = (params.groqApiKey ?? savedCfg.groqKey ?? '').trim();
  const effectiveOpenRouterKey = (params.openRouterApiKey ?? savedCfg.openRouterKey ?? '').trim();
  const effectiveProviderPref: AiProviderPref = params.providerPref || savedCfg.providerPref || 'auto';
  const effectivePreferredGemini =
    params.preferredGeminiModel && params.preferredGeminiModel !== 'auto'
      ? params.preferredGeminiModel
      : savedCfg.preferredGeminiModel !== 'auto'
      ? savedCfg.preferredGeminiModel
      : '';

  const aiClient = effectiveGeminiKey ? new GoogleGenAI({ apiKey: effectiveGeminiKey }) : null;
  const discoveredModels = effectiveGeminiKey
    ? await discoverFlashModelsDirect(effectiveGeminiKey)
    : [];

  const JSON_SCHEMA_HINT = `
RESPONDA ESTRITAMENTE COM UM ÚNICO OBJETO JSON VÁLIDO (sem markdown, sem comentários) seguindo exatamente esta estrutura:
{
  "estimatedTotalScore": number,
  "totalTimeFormatted": "MM:SS",
  "totalTimeSeconds": number,
  "overallConfidence": number (0 a 100),
  "summary": "resumo curto em português",
  "trajectoryNotes": "observações de trajetória",
  "mechanismNotes": "observações de mecanismos",
  "missionsDetected": [
    {
      "missionCode": "M01",
      "status": "Completa" | "Parcial" | "Incompleta",
      "estimatedScore": number,
      "timestampEstimate": "00:12",
      "evidenceDescription": "o que foi visto no quadro/vídeo",
      "confidence": number (0 a 100)
    }
  ]
}`;

  const runOpenAiCompatibleVision = async (
    provider: 'groq' | 'openrouter',
    providerKey: string,
    mediaParts: any[],
    promptText: string,
    progressPct: number,
    stepLabel: string
  ): Promise<{ parsedData: any; successfulModel: string }> => {
    const endpoint =
      provider === 'groq'
        ? 'https://api.groq.com/openai/v1/chat/completions'
        : 'https://openrouter.ai/api/v1/chat/completions';

    const modelsToTry =
      provider === 'groq'
        ? [
            'meta-llama/llama-4-scout-17b-16e-instruct',
            'meta-llama/llama-4-maverick-17b-128e-instruct',
          ]
        : [
            'google/gemma-3-27b-it:free',
            'qwen/qwen2.5-vl-72b-instruct:free',
            'meta-llama/llama-4-scout:free',
            'openrouter/free',
          ];

    // Keep total images compact so Groq (max ~5 images/req on some models) & OpenRouter accept payloads reliably
    const contentItems: any[] = [];
    const maxRefImagesForThirdParty = provider === 'groq' ? 2 : 4;
    let refImgsAdded = 0;

    contentItems.push({
      type: 'text',
      text: `Você é um árbitro rigoroso da FLL (BIOGLOW). Códigos permitidos: [${allowedCodesList.join(', ')}].\n${JSON_SCHEMA_HINT}`,
    });

    for (const p of [...referenceParts, ...trainingParts]) {
      if (p?.text) {
        contentItems.push({ type: 'text', text: p.text });
      } else if (p?.inlineData?.data && refImgsAdded < maxRefImagesForThirdParty) {
        refImgsAdded++;
        contentItems.push({
          type: 'image_url',
          image_url: {
            url: `data:${p.inlineData.mimeType || 'image/jpeg'};base64,${p.inlineData.data}`,
          },
        });
      }
    }

    contentItems.push({ type: 'text', text: 'MÍDIA DO ROUND:' });
    let mediaImgsAdded = 0;
    const maxMediaImgs = provider === 'groq' ? 3 : 8;
    const imageMediaParts = mediaParts.filter((p) => p?.inlineData?.data);
    // Evenly sample media frames if provider has an image-count cap
    const stepSample =
      imageMediaParts.length > maxMediaImgs ? imageMediaParts.length / maxMediaImgs : 1;

    if (imageMediaParts.length > maxMediaImgs) {
      for (let k = 0; k < maxMediaImgs; k++) {
        const idx = Math.min(imageMediaParts.length - 1, Math.floor(k * stepSample));
        const imgP = imageMediaParts[idx];
        contentItems.push({
          type: 'image_url',
          image_url: {
            url: `data:${imgP.inlineData.mimeType || 'image/jpeg'};base64,${imgP.inlineData.data}`,
          },
        });
      }
    } else {
      for (const p of mediaParts) {
        if (p?.text) {
          contentItems.push({ type: 'text', text: p.text });
        } else if (p?.inlineData?.data && mediaImgsAdded < maxMediaImgs) {
          mediaImgsAdded++;
          contentItems.push({
            type: 'image_url',
            image_url: {
              url: `data:${p.inlineData.mimeType || 'image/jpeg'};base64,${p.inlineData.data}`,
            },
          });
        }
      }
    }

    contentItems.push({ type: 'text', text: promptText });

    let lastThirdPartyErr: any = null;
    for (let i = 0; i < modelsToTry.length; i++) {
      if (signal?.aborted) throw new Error('ABORTED');
      const modelId = modelsToTry[i];
      onProgress(
        Math.min(95, progressPct + i * 2),
        `${stepLabel} via ${provider === 'groq' ? 'Groq' : 'OpenRouter'} (${modelId})...`
      );

      const ctrl = new AbortController();
      const timer = window.setTimeout(() => ctrl.abort(), 45000);
      try {
        const headers: Record<string, string> = {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${providerKey}`,
        };
        if (provider === 'openrouter') {
          headers['HTTP-Referer'] = window.location.origin;
          headers['X-Title'] = 'FrancoDroid FLL Analytics';
        }

        const res = await fetch(endpoint, {
          method: 'POST',
          headers,
          signal: ctrl.signal,
          body: JSON.stringify({
            model: modelId,
            messages: [{ role: 'user', content: contentItems }],
            temperature: 0.1,
            max_tokens: 1800,
          }),
        });
        window.clearTimeout(timer);

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(data?.error?.message || `HTTP ${res.status}`);
        }
        const rawText = String(data?.choices?.[0]?.message?.content || '');
        const parsedData = extractJsonObjectFromText(rawText);
        return {
          parsedData,
          successfulModel: `${provider}:${modelId}`,
        };
      } catch (err: any) {
        window.clearTimeout(timer);
        lastThirdPartyErr = err;
        const msg = String(err?.message || '').toLowerCase();
        if (msg.includes('401') || msg.includes('invalid api key') || msg.includes('unauthorized')) {
          throw new Error(
            `Chave ${provider === 'groq' ? 'Groq' : 'OpenRouter'} inválida. Confira na aba "Configurar IA".`
          );
        }
      }
    }

    throw new Error(
      `Falha no provedor alternativo ${provider === 'groq' ? 'Groq' : 'OpenRouter'}: ${String(
        lastThirdPartyErr?.message || 'servidor ocupado'
      )}`
    );
  };

  const runGeminiDirect = async (
    mediaParts: any[],
    promptText: string,
    progressPct: number,
    stepLabel: string,
    opts: { isMultiPart?: boolean; pinnedModel?: string; hasImageFramesOnly?: boolean } = {}
  ): Promise<{ parsedData: any; successfulModel: string }> => {
    // Handle direct third-party preference when caller passed frames/image
    const canSendToThirdParty =
      opts.hasImageFramesOnly ||
      mediaParts.every((p) => !p?.inlineData?.mimeType?.startsWith('video/'));

    if (effectiveProviderPref === 'groq' && effectiveGroqKey && canSendToThirdParty) {
      return runOpenAiCompatibleVision('groq', effectiveGroqKey, mediaParts, promptText, progressPct, stepLabel);
    }
    if (effectiveProviderPref === 'openrouter' && effectiveOpenRouterKey && canSendToThirdParty) {
      return runOpenAiCompatibleVision(
        'openrouter',
        effectiveOpenRouterKey,
        mediaParts,
        promptText,
        progressPct,
        stepLabel
      );
    }

    if (!aiClient) {
      if (effectiveGroqKey && canSendToThirdParty) {
        return runOpenAiCompatibleVision('groq', effectiveGroqKey, mediaParts, promptText, progressPct, stepLabel);
      }
      if (effectiveOpenRouterKey && canSendToThirdParty) {
        return runOpenAiCompatibleVision(
          'openrouter',
          effectiveOpenRouterKey,
          mediaParts,
          promptText,
          progressPct,
          stepLabel
        );
      }
      throw new Error('Nenhuma chave de IA configurada. Adicione uma chave Gemini, Groq ou OpenRouter em "Configurar IA".');
    }

    const startPool = opts.pinnedModel
      ? [opts.pinnedModel, ...discoveredModels.filter((m) => m !== opts.pinnedModel)]
      : effectivePreferredGemini
      ? [effectivePreferredGemini, ...discoveredModels.filter((m) => m !== effectivePreferredGemini)]
      : discoveredModels;

    // Gemma text/vision models cannot parse raw MP4 inline video parts (only images)
    const isVideoPayload = mediaParts.some((p) =>
      String(p?.inlineData?.mimeType || '').startsWith('video/')
    );
    const filteredPool = isVideoPayload
      ? startPool.filter((m) => !m.startsWith('gemma-'))
      : startPool;
    const baseModels = Array.from(new Set(filteredPool)).slice(0, 6);

    const multimodalParts = [
      ...referenceParts,
      ...trainingParts,
      { text: 'MÍDIA DO ROUND:' },
      ...mediaParts,
      { text: promptText },
    ];

    const perModelTimeoutMs = opts.isMultiPart ? 40000 : 52000;
    let lastErr: any = null;
    let sawTransient503 = false;

    const trySingleModel = async (modelCandidate: string, pct: number, label: string) => {
      onProgress(Math.min(96, pct), label);
      const isGemma = modelCandidate.startsWith('gemma-');

      const buildCfg = (includeAdvanced: boolean) => {
        if (isGemma) {
          return { temperature: 0.1 };
        }
        const cfg: any = {
          systemInstruction: `Você é um árbitro rigoroso da FLL (BIOGLOW). Retorne estritamente JSON válido. Nunca invente missões não executadas no vídeo. Códigos permitidos: [${allowedCodesList.join(', ')}].`,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              estimatedTotalScore: { type: Type.NUMBER },
              totalTimeFormatted: { type: Type.STRING },
              totalTimeSeconds: { type: Type.NUMBER },
              overallConfidence: { type: Type.NUMBER },
              summary: { type: Type.STRING },
              trajectoryNotes: { type: Type.STRING },
              mechanismNotes: { type: Type.STRING },
              missionsDetected: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    missionCode: { type: Type.STRING },
                    status: { type: Type.STRING },
                    estimatedScore: { type: Type.NUMBER },
                    timestampEstimate: { type: Type.STRING },
                    evidenceDescription: { type: Type.STRING },
                    confidence: { type: Type.NUMBER },
                  },
                  required: ['missionCode', 'status', 'estimatedScore', 'evidenceDescription'],
                },
              },
            },
            required: [
              'estimatedTotalScore',
              'totalTimeFormatted',
              'totalTimeSeconds',
              'overallConfidence',
              'summary',
              'trajectoryNotes',
              'mechanismNotes',
              'missionsDetected',
            ],
          },
        };
        if (includeAdvanced) {
          const targetRes = opts.isMultiPart
            ? MediaResolution.MEDIA_RESOLUTION_LOW
            : MediaResolution.MEDIA_RESOLUTION_MEDIUM;
          if (modelCandidate === 'gemini-4-hydrogen' || modelCandidate === 'gemini-3.8-flash') {
            cfg.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
            cfg.mediaResolution = targetRes;
          } else if (modelCandidate === 'gemini-3.1-flash-lite') {
            cfg.thinkingConfig = { thinkingLevel: ThinkingLevel.MINIMAL };
            cfg.mediaResolution = targetRes;
          }
        }
        return cfg;
      };

      const partsForModel = isGemma
        ? [
            ...referenceParts,
            ...trainingParts,
            { text: 'MÍDIA DO ROUND:' },
            ...mediaParts,
            {
              text: `Você é um árbitro rigoroso da FLL (BIOGLOW). Códigos permitidos: [${allowedCodesList.join(', ')}].\n${promptText}\n${JSON_SCHEMA_HINT}`,
            },
          ]
        : multimodalParts;

      const timeoutPromise = new Promise((_, reject) =>
        window.setTimeout(
          () =>
            reject(
              new Error(
                `Tempo limite (${Math.round(perModelTimeoutMs / 1000)}s) no modelo ${modelCandidate}`
              )
            ),
          perModelTimeoutMs
        )
      );

      let response: any;
      try {
        response = await Promise.race([
          aiClient.models.generateContent({
            model: modelCandidate,
            contents: { parts: partsForModel },
            config: buildCfg(true),
          }),
          timeoutPromise,
        ]);
      } catch (firstErr: any) {
        const msg = String(firstErr?.message || '');
        if (
          !isGemma &&
          (msg.includes('INVALID_ARGUMENT') ||
            msg.includes('400') ||
            msg.toLowerCase().includes('thinking') ||
            msg.toLowerCase().includes('mediaresolution'))
        ) {
          response = await Promise.race([
            aiClient.models.generateContent({
              model: modelCandidate,
              contents: { parts: partsForModel },
              config: buildCfg(false),
            }),
            timeoutPromise,
          ]);
        } else {
          throw firstErr;
        }
      }

      const parsedData = extractJsonObjectFromText(response.text ?? '{}');
      return { parsedData, successfulModel: modelCandidate };
    };

    for (let i = 0; i < baseModels.length; i++) {
      if (signal?.aborted) throw new Error('ABORTED');
      const modelCandidate = baseModels[i];
      try {
        return await trySingleModel(
          modelCandidate,
          progressPct + i * 2,
          `${stepLabel} (${modelCandidate})...`
        );
      } catch (err: any) {
        lastErr = err;
        const rawMsg = extractGeminiErrorText(err);
        if (rawMsg.includes('API_KEY_INVALID') || rawMsg.includes('PERMISSION_DENIED')) break;

        if (isTransientOverloadError(err)) {
          sawTransient503 = true;
          if (i < baseModels.length - 1) {
            onProgress(
              Math.min(95, progressPct + i * 2 + 1),
              `Modelo ${modelCandidate} ocupado (503) — alternando para ${baseModels[i + 1]}...`
            );
            await new Promise((r) => window.setTimeout(r, 1000));
          }
        }
      }
    }

    // Try automatic fallback to Groq / OpenRouter if Gemini hit 503 and frame images are being analyzed
    if (
      sawTransient503 &&
      effectiveProviderPref === 'auto' &&
      canSendToThirdParty &&
      !signal?.aborted
    ) {
      if (effectiveGroqKey) {
        try {
          onProgress(Math.min(95, progressPct + 4), 'Gemini ocupado (503) — alternando para Groq Vision...');
          return await runOpenAiCompatibleVision(
            'groq',
            effectiveGroqKey,
            mediaParts,
            promptText,
            progressPct + 4,
            stepLabel
          );
        } catch {
          // continue to openrouter or backoff retry
        }
      }
      if (effectiveOpenRouterKey) {
        try {
          onProgress(
            Math.min(95, progressPct + 4),
            'Gemini ocupado (503) — alternando para OpenRouter Vision...'
          );
          return await runOpenAiCompatibleVision(
            'openrouter',
            effectiveOpenRouterKey,
            mediaParts,
            promptText,
            progressPct + 4,
            stepLabel
          );
        } catch {
          // continue to backoff retry
        }
      }
    }

    // One brief jittered backoff retry round if everything hit transient 503/high demand
    if (sawTransient503 && !signal?.aborted && baseModels.length > 0) {
      const retryQueue = [...baseModels]
        .sort((a, b) => {
          const score = (m: string) => (m.startsWith('gemma-') ? 0 : m.includes('lite') ? 1 : 2);
          return score(a) - score(b);
        })
        .slice(0, 2);

      onProgress(
        Math.min(95, progressPct + 4),
        'Servidores Gemini com pico temporário (503). Aguardando 2s para nova tentativa...'
      );
      await new Promise((r) => window.setTimeout(r, 2000));

      for (let j = 0; j < retryQueue.length; j++) {
        if (signal?.aborted) throw new Error('ABORTED');
        const retryModel = retryQueue[j];
        try {
          return await trySingleModel(
            retryModel,
            progressPct + 6 + j,
            `Tentando novamente (${retryModel})...`
          );
        } catch (retryErr: any) {
          lastErr = retryErr;
          if (j < retryQueue.length - 1 && isTransientOverloadError(retryErr)) {
            await new Promise((r) => window.setTimeout(r, 1200));
          }
        }
      }
    }

    throw new Error(formatClientGeminiError(lastErr));
  };

  const allowedSanitizeList = activeMissions.map((m) => ({
    code: m.code,
    maxScore: m.maxScore || 30,
    validScores: m.validScores,
  }));

  if (!isVideo) {
    onProgress(35, 'Preparando foto da mesa...');
    const b64 = await fileToBase64Raw(mediaFile);
    const { parsedData, successfulModel } = await runGeminiDirect(
      [{ inlineData: { mimeType: mediaFile.type || 'image/jpeg', data: b64 } }],
      buildPrompt(),
      55,
      'Analisando foto com IA',
      { hasImageFramesOnly: true }
    );
    return {
      ...sanitizeClientAnalysisOutput(
        parsedData,
        allowedSanitizeList,
        hasGabarito,
        relevantTraining.length,
        cleanedInformed
      ),
      modelUsed: successfulModel,
    };
  }

  // Video mode (browser canvas keyframes + direct fallback)
  onProgress(22, 'Verificando duração do vídeo no navegador...');
  const probedDur =
    params.videoDuration > 0 ? params.videoDuration : await probeBrowserVideoDuration(mediaFile);
  const effectiveDuration = Math.max(2, probedDur || 30);
  const fileMb = mediaFile.size / (1024 * 1024);

  const usingThirdPartyDirect =
    effectiveProviderPref === 'groq' ||
    effectiveProviderPref === 'openrouter' ||
    (!effectiveGeminiKey && (Boolean(effectiveGroqKey) || Boolean(effectiveOpenRouterKey)));

  const shouldUseParts =
    Boolean(params.forceKeyframes) ||
    usingThirdPartyDirect ||
    splitMode === 'parts' ||
    fileMb > 12 ||
    (splitMode === 'auto' && effectiveDuration > 40);

  if (!shouldUseParts && fileMb <= 12) {
    try {
      onProgress(40, 'Preparando vídeo direto para IA...');
      const b64 = await fileToBase64Raw(mediaFile);
      const { parsedData, successfulModel } = await runGeminiDirect(
        [{ inlineData: { mimeType: mediaFile.type || 'video/mp4', data: b64 } }],
        buildPrompt(),
        60,
        'Analisando vídeo com IA'
      );
      return {
        ...sanitizeClientAnalysisOutput(
          parsedData,
          allowedSanitizeList,
          hasGabarito,
          relevantTraining.length,
          cleanedInformed
        ),
        modelUsed: successfulModel,
      };
    } catch (directErr: any) {
      if (String(directErr?.message || '') === 'ABORTED') throw directErr;
      // On 503/high demand for raw MP4 upload, fall through automatically to lightweight JPEG keyframes!
      if (!isTransientOverloadError(directErr)) {
        throw directErr;
      }
      onProgress(
        35,
        'Envio MP4 congestionado (503). Extraindo quadros leves anti-503 automaticamente...'
      );
    }
  }

  // Split into <= 4 segments (or 1 compact segment for <= 20s forced keyframes) and extract crisp JPEG frames
  const maxParts = 4;
  const partCount =
    effectiveDuration <= 18 && params.forceKeyframes
      ? 1
      : effectiveDuration <= 22
      ? 2
      : Math.min(maxParts, Math.max(2, Math.ceil(effectiveDuration / 30)));
  const segLen = effectiveDuration / partCount;
  const segments: Array<{ index: number; startSec: number; endSec: number }> = [];
  for (let i = 0; i < partCount; i++) {
    const startSec = Math.max(0, i * segLen - (i > 0 ? 1.5 : 0));
    const endSec = Math.min(effectiveDuration, (i + 1) * segLen);
    segments.push({ index: i, startSec, endSec });
  }

  const partResults: Array<{ startSec: number; endSec: number; sanitized: any }> = [];
  let pinnedModel = '';
  let partialWarning = '';

  const runSegment = async (seg: { index: number; startSec: number; endSec: number }, modelPin?: string) => {
    const label = `${formatSecShort(seg.startSec)}–${formatSecShort(seg.endSec)}`;
    const basePct = 28 + Math.round((seg.index / segments.length) * 62);
    onProgress(basePct, `Extraindo quadros Parte ${seg.index + 1}/${segments.length} (${label})...`);

    const frames = await extractVideoSegmentKeyframes(mediaFile, seg.startSec, seg.endSec, 9, signal);
    const frameParts: any[] = [];
    for (const f of frames) {
      frameParts.push({ text: `[Quadro em ${formatSecShort(f.timestampSec)}]` });
      frameParts.push({ inlineData: { mimeType: 'image/jpeg', data: f.base64Data } });
    }

    const promptText = buildPrompt(
      `Parte ${seg.index + 1}/${segments.length} (${label} de ${formatSecShort(effectiveDuration)}). Avalie apenas ações visíveis nestes quadros.`
    );

    const { parsedData, successfulModel } = await runGeminiDirect(
      frameParts,
      promptText,
      basePct + 6,
      `Analisando Parte ${seg.index + 1}/${segments.length} (${label})`,
      { isMultiPart: true, pinnedModel: modelPin, hasImageFramesOnly: true }
    );

    return {
      startSec: seg.startSec,
      endSec: seg.endSec,
      successfulModel,
      sanitized: sanitizeClientAnalysisOutput(
        parsedData,
        allowedSanitizeList,
        hasGabarito,
        relevantTraining.length,
        cleanedInformed
      ),
    };
  };

  // Part 1 discovers & pins working model
  const first = await runSegment(segments[0]);
  pinnedModel = first.successfulModel;
  partResults.push({ startSec: first.startSec, endSec: first.endSec, sanitized: first.sanitized });

  for (let i = 1; i < segments.length; i += 2) {
    if (signal?.aborted) throw new Error('ABORTED');
    const batch = segments.slice(i, i + 2);
    const settled = await Promise.allSettled(batch.map((s) => runSegment(s, pinnedModel)));

    let batchErr: any = null;
    for (const outcome of settled) {
      if (outcome.status === 'fulfilled') {
        const v = outcome.value;
        partResults.push({ startSec: v.startSec, endSec: v.endSec, sanitized: v.sanitized });
      } else if (!batchErr) {
        batchErr = outcome.reason;
      }
    }
    if (batchErr) {
      const msg = String(batchErr?.message || '');
      if (msg === 'ABORTED') throw batchErr;
      partialWarning = `Partes analisadas: ${partResults.length}/${segments.length}. Uma parte seguinte foi interrompida: ${formatClientGeminiError(batchErr)}`;
      break;
    }
  }

  const merged = mergeClientPartAnalyses(
    partResults,
    effectiveDuration,
    hasGabarito,
    relevantTraining.length
  );
  if (partialWarning) {
    merged.gabaritoWarning = partialWarning;
  }

  return {
    ...merged,
    modelUsed: pinnedModel,
  };
}
