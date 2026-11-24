import {
  User,
  onAuthStateChanged,
  signInWithPopup,
  signOut,
} from 'firebase/auth';
import {
  collection,
  deleteDoc,
  doc,
  getDocs,
  onSnapshot,
  query,
  serverTimestamp,
  setDoc,
  where,
} from 'firebase/firestore';
import {
  auth,
  db,
  googleProvider,
  handleFirestoreError,
  OperationType,
} from '../firebase';
import { AiTrainingExample, Round } from '../types/fll';
import { StorageService } from './storageService';
import { hasBackendServer } from './clientAiEngine';

const VALID_MISTAKE_TYPES = new Set([
  'false_positive',
  'wrong_score',
  'missed_mission',
  'confused_mission',
  'other',
]);

// Track actual Firestore document IDs by logical local ID so updates/deletes hit the exact doc
const cloudRoundDocIdByLogicalId = new Map<string, string>();
const cloudTrainingDocIdByLogicalId = new Map<string, string>();

export function toSafeFirestoreId(rawId: string, prefix = 'item'): string {
  const cleaned = String(rawId || '')
    .replace(/[^a-zA-Z0-9_\-]/g, '_')
    .slice(0, 128);
  return cleaned.length > 0 ? cleaned : `${prefix}_${Date.now()}`;
}

function getUidPrefix(uid: string): string {
  return toSafeFirestoreId(uid, 'u').slice(0, 10);
}

function toUserScopedDocId(rawId: string, uid: string, prefix = 'item'): string {
  const safeBase = toSafeFirestoreId(rawId, prefix);
  const uidPrefix = getUidPrefix(uid);
  const scoped = safeBase.startsWith(`${uidPrefix}_`) ? safeBase : `${uidPrefix}_${safeBase}`;
  return scoped.slice(0, 128);
}

function toLogicalIdFromCloudId(cloudId: string, uid: string): string {
  const uidPrefix = `${getUidPrefix(uid)}_`;
  return cloudId.startsWith(uidPrefix) ? cloudId.slice(uidPrefix.length) : cloudId;
}

function buildFirestoreRoundPayload(round: Round, ownerId: string, docId: string) {
  const safeMissionResults = Array.isArray(round.missionResults)
    ? round.missionResults.slice(0, 20).map((mr) => ({
        missionId: String(mr.missionId || `m_${mr.missionCode || 'x'}`),
        missionCode: String(mr.missionCode || 'M01').slice(0, 16),
        status: String(mr.status || 'completed'),
        score: Math.max(0, Number(mr.score) || 0),
        attempts: Math.max(1, Number(mr.attempts) || 1),
        timeSpentSeconds: Math.max(0, Number(mr.timeSpentSeconds) || 0),
        notes: mr.notes ? String(mr.notes).slice(0, 500) : '',
      }))
    : [];

  return {
    id: docId,
    ownerId,
    roundNumber: Math.max(0, Math.min(100000, Number(round.roundNumber) || 0)),
    date: String(round.date || new Date().toISOString().slice(0, 16)).slice(0, 64),
    operator: String(round.operator || 'Piloto FrancoDroid').slice(0, 100),
    strategy: String(round.strategy || 'Análise IA').slice(0, 160),
    totalTimeSeconds: Math.max(0, Math.min(600, Number(round.totalTimeSeconds) || 0)),
    totalScore: Math.max(0, Math.min(1000, Number(round.totalScore) || 0)),
    notes: String(round.notes || '').slice(0, 2500),
    missionResults: safeMissionResults,
    recordedAtMs: Math.max(0, Number(round.createdAt) || Date.now()),
    verifiedByAi: Boolean(round.verifiedByAi),
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };
}

function buildFirestoreTrainingPayload(
  ex: AiTrainingExample,
  ownerId: string,
  docId: string
) {
  const mistakeType = VALID_MISTAKE_TYPES.has(ex.mistakeType) ? ex.mistakeType : 'other';

  const basePayload: Record<string, any> = {
    id: docId,
    ownerId,
    missionCode: String(ex.missionCode || 'GERAL').toUpperCase().slice(0, 16),
    mistakeType,
    aiPredictedSummary: String(ex.aiPredictedSummary || '').slice(0, 500),
    expectedSummary: String(ex.expectedSummary || '').slice(0, 500),
    userExplanation: String(ex.userExplanation || '').slice(0, 1000),
    ruleCreatedAtIso: String(ex.createdAt || new Date().toISOString()).slice(0, 64),
    createdAt: serverTimestamp(),
    updatedAt: serverTimestamp(),
  };

  if (
    typeof ex.snapshotBase64 === 'string' &&
    ex.snapshotBase64.length > 0 &&
    ex.snapshotBase64.length <= 360000
  ) {
    basePayload.snapshotBase64 = ex.snapshotBase64;
  }

  return basePayload;
}

function mapFirestoreDocToRound(data: any, uid: string): Round | null {
  if (!data || !data.id) return null;
  const cloudId = String(data.id);
  const logicalId = toLogicalIdFromCloudId(cloudId, uid);
  cloudRoundDocIdByLogicalId.set(logicalId, cloudId);
  return {
    id: logicalId,
    roundNumber: Number(data.roundNumber) || 1,
    date: String(data.date || ''),
    operator: String(data.operator || 'Piloto FrancoDroid'),
    strategy: String(data.strategy || 'Análise IA'),
    totalTimeSeconds: Number(data.totalTimeSeconds) || 0,
    totalScore: Number(data.totalScore) || 0,
    notes: String(data.notes || ''),
    missionResults: Array.isArray(data.missionResults) ? data.missionResults : [],
    createdAt: Number(data.recordedAtMs) || Date.now(),
    verifiedByAi: Boolean(data.verifiedByAi),
  };
}

function mapFirestoreDocToTraining(data: any, uid: string): AiTrainingExample | null {
  if (!data || !data.id) return null;
  const cloudId = String(data.id);
  const logicalId = toLogicalIdFromCloudId(cloudId, uid);
  cloudTrainingDocIdByLogicalId.set(logicalId, cloudId);
  return {
    id: logicalId,
    createdAt: String(data.ruleCreatedAtIso || new Date().toISOString()),
    missionCode: String(data.missionCode || 'GERAL'),
    mistakeType: VALID_MISTAKE_TYPES.has(data.mistakeType) ? data.mistakeType : 'other',
    aiPredictedSummary: String(data.aiPredictedSummary || ''),
    expectedSummary: String(data.expectedSummary || ''),
    userExplanation: String(data.userExplanation || ''),
    snapshotBase64: typeof data.snapshotBase64 === 'string' ? data.snapshotBase64 : undefined,
  };
}

export const CloudSyncService = {
  subscribeAuth(callback: (user: User | null) => void) {
    return onAuthStateChanged(auth, callback);
  },

  async signInWithGoogle(): Promise<User> {
    const result = await signInWithPopup(auth, googleProvider);
    return result.user;
  },

  async signOutCloud(): Promise<void> {
    cloudRoundDocIdByLogicalId.clear();
    cloudTrainingDocIdByLogicalId.clear();
    await signOut(auth);
  },

  async pushRoundsToServer(rounds: Round[], deletedIds: string[] = []): Promise<Round[]> {
    try {
      if (!(await hasBackendServer())) return rounds;
      const res = await fetch('./api/rounds', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rounds, deletedIds }),
      });
      if (!res.ok) return rounds;
      const data = await res.json();
      return Array.isArray(data?.rounds) ? data.rounds : rounds;
    } catch {
      return rounds;
    }
  },

  async pushTrainingToServer(
    examples: AiTrainingExample[],
    deletedIds: string[] = []
  ): Promise<AiTrainingExample[]> {
    try {
      if (!(await hasBackendServer())) return examples;
      const res = await fetch('./api/ai/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ examples, deletedIds, syncOnly: true }),
      });
      if (!res.ok) return examples;
      const data = await res.json();
      return Array.isArray(data?.examples) ? data.examples : examples;
    } catch {
      return examples;
    }
  },

  /**
   * Non-destructive startup sync between browser localStorage and server disk JSON.
   * Ensures neither resets the other even after deploys or reloads.
   */
  async bootstrapServerSync(): Promise<{
    rounds: Round[];
    training: AiTrainingExample[];
  }> {
    const localRounds = StorageService.getRounds();
    const localTraining = StorageService.getAiTraining();
    const deletedRounds = StorageService.getDeletedRoundIds();
    const deletedTraining = StorageService.getDeletedTrainingIds();

    if (!(await hasBackendServer())) {
      return { rounds: localRounds, training: localTraining };
    }

    let finalRounds = localRounds;
    let finalTraining = localTraining;

    try {
      const [roundsRes, trainRes] = await Promise.all([
        fetch('./api/rounds'),
        fetch('./api/ai/training'),
      ]);

      if (roundsRes.ok) {
        const rd = await roundsRes.json();
        const serverRounds: Round[] = Array.isArray(rd?.rounds) ? rd.rounds : [];
        finalRounds = StorageService.mergeRounds(localRounds, serverRounds);
        StorageService.saveRounds(finalRounds);
        if (finalRounds.length > serverRounds.length || deletedRounds.length > 0) {
          finalRounds = await this.pushRoundsToServer(finalRounds, deletedRounds);
          finalRounds = StorageService.mergeRounds(finalRounds, []);
          StorageService.saveRounds(finalRounds);
        }
      }

      if (trainRes.ok) {
        const td = await trainRes.json();
        const serverTraining: AiTrainingExample[] = Array.isArray(td?.examples) ? td.examples : [];
        finalTraining = StorageService.mergeAiTraining(localTraining, serverTraining);
        StorageService.saveAiTraining(finalTraining);
        if (finalTraining.length > serverTraining.length || deletedTraining.length > 0) {
          finalTraining = await this.pushTrainingToServer(finalTraining, deletedTraining);
          finalTraining = StorageService.mergeAiTraining(finalTraining, []);
          StorageService.saveAiTraining(finalTraining);
        }
      }
    } catch (err) {
      console.warn('Falha ao sincronizar com servidor local:', err);
    }

    return { rounds: finalRounds, training: finalTraining };
  },

  async saveRoundEverywhere(round: Round, user: User | null): Promise<{
    updated: Round[];
    newRound: Round;
  }> {
    const { updated, newRound } = StorageService.addRound(round);
    await this.pushRoundsToServer([newRound], StorageService.getDeletedRoundIds());

    if (user && user.emailVerified) {
      const docId = toUserScopedDocId(newRound.id, user.uid, 'round');
      const path = `rounds/${docId}`;
      try {
        await setDoc(
          doc(db, 'rounds', docId),
          buildFirestoreRoundPayload(newRound, user.uid, docId)
        );
        cloudRoundDocIdByLogicalId.set(newRound.id, docId);
      } catch (err) {
        handleFirestoreError(err, OperationType.CREATE, path);
      }
    }

    return { updated, newRound };
  },

  async deleteRoundEverywhere(id: string, user: User | null): Promise<Round[]> {
    const updated = StorageService.deleteRound(id);
    if (await hasBackendServer()) {
      fetch(`./api/rounds/${encodeURIComponent(id)}`, { method: 'DELETE' }).catch(() => {});
    }

    if (user && user.emailVerified) {
      const trackedDocId =
        cloudRoundDocIdByLogicalId.get(id) || toUserScopedDocId(id, user.uid, 'round');
      const path = `rounds/${trackedDocId}`;
      try {
        await deleteDoc(doc(db, 'rounds', trackedDocId));
        cloudRoundDocIdByLogicalId.delete(id);
      } catch (err) {
        handleFirestoreError(err, OperationType.DELETE, path);
      }
    }

    return updated;
  },

  async saveTrainingEverywhere(
    newExamples: Array<Omit<AiTrainingExample, 'id' | 'createdAt'>>,
    user: User | null
  ): Promise<AiTrainingExample[]> {
    let incomingExamples: AiTrainingExample[] = newExamples.map((ex, idx) => ({
      id: `train_${Date.now()}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
      createdAt: new Date().toISOString(),
      missionCode: String(ex.missionCode || 'GERAL').toUpperCase(),
      mistakeType: VALID_MISTAKE_TYPES.has(ex.mistakeType) ? ex.mistakeType : 'other',
      aiPredictedSummary: String(ex.aiPredictedSummary || ''),
      expectedSummary: String(ex.expectedSummary || ''),
      userExplanation: String(ex.userExplanation || ''),
      snapshotBase64: ex.snapshotBase64,
    }));

    if (await hasBackendServer()) {
      const res = await fetch('./api/ai/training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ examples: incomingExamples }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error || 'Falha ao salvar correção.');
      }
    }

    const merged = StorageService.mergeAiTraining(StorageService.getAiTraining(), incomingExamples);
    StorageService.saveAiTraining(merged);

    if (user && user.emailVerified) {
      for (const item of incomingExamples) {
        const existingDocId = cloudTrainingDocIdByLogicalId.get(item.id);
        if (existingDocId) continue; // Already stored in Firestore with immutable createdAt
        const docId = toUserScopedDocId(item.id, user.uid, 'train');
        const path = `aiTraining/${docId}`;
        try {
          await setDoc(
            doc(db, 'aiTraining', docId),
            buildFirestoreTrainingPayload(item, user.uid, docId)
          );
          cloudTrainingDocIdByLogicalId.set(item.id, docId);
        } catch (err) {
          handleFirestoreError(err, OperationType.CREATE, path);
        }
      }
    }

    return merged;
  },

  async deleteTrainingEverywhere(id: string, user: User | null): Promise<AiTrainingExample[]> {
    const localUpdated = StorageService.deleteAiTraining(id);
    if (await hasBackendServer()) {
      try {
        await fetch(`./api/ai/training/${encodeURIComponent(id)}`, { method: 'DELETE' });
      } catch {}
    }

    if (user && user.emailVerified) {
      const trackedDocId =
        cloudTrainingDocIdByLogicalId.get(id) || toUserScopedDocId(id, user.uid, 'train');
      const path = `aiTraining/${trackedDocId}`;
      try {
        await deleteDoc(doc(db, 'aiTraining', trackedDocId));
        cloudTrainingDocIdByLogicalId.delete(id);
      } catch (err) {
        handleFirestoreError(err, OperationType.DELETE, path);
      }
    }

    return localUpdated;
  },

  async pushAllLocalToCloud(
    rounds: Round[],
    training: AiTrainingExample[],
    user: User
  ): Promise<void> {
    if (!user.emailVerified) return;

    // Fetch existing user-owned docs first so we never re-run create on an existing document
    const existingRoundLogicalIds = new Set<string>();
    const existingTrainingLogicalIds = new Set<string>();

    try {
      const rSnap = await getDocs(
        query(collection(db, 'rounds'), where('ownerId', '==', user.uid))
      );
      rSnap.docs.forEach((d) => {
        const cloudId = String(d.data()?.id || d.id);
        const logicalId = toLogicalIdFromCloudId(cloudId, user.uid);
        existingRoundLogicalIds.add(logicalId);
        cloudRoundDocIdByLogicalId.set(logicalId, cloudId);
      });
    } catch (err) {
      handleFirestoreError(err, OperationType.LIST, 'rounds');
    }

    try {
      const tSnap = await getDocs(
        query(collection(db, 'aiTraining'), where('ownerId', '==', user.uid))
      );
      tSnap.docs.forEach((d) => {
        const cloudId = String(d.data()?.id || d.id);
        const logicalId = toLogicalIdFromCloudId(cloudId, user.uid);
        existingTrainingLogicalIds.add(logicalId);
        cloudTrainingDocIdByLogicalId.set(logicalId, cloudId);
      });
    } catch (err) {
      handleFirestoreError(err, OperationType.LIST, 'aiTraining');
    }

    for (const r of rounds) {
      if (existingRoundLogicalIds.has(r.id) || existingRoundLogicalIds.has(toSafeFirestoreId(r.id, 'round'))) {
        continue;
      }
      const docId = toUserScopedDocId(r.id, user.uid, 'round');
      const path = `rounds/${docId}`;
      try {
        await setDoc(doc(db, 'rounds', docId), buildFirestoreRoundPayload(r, user.uid, docId));
        cloudRoundDocIdByLogicalId.set(r.id, docId);
      } catch (err) {
        handleFirestoreError(err, OperationType.CREATE, path);
      }
    }

    for (const t of training) {
      if (
        existingTrainingLogicalIds.has(t.id) ||
        existingTrainingLogicalIds.has(toSafeFirestoreId(t.id, 'train'))
      ) {
        continue;
      }
      const docId = toUserScopedDocId(t.id, user.uid, 'train');
      const path = `aiTraining/${docId}`;
      try {
        await setDoc(doc(db, 'aiTraining', docId), buildFirestoreTrainingPayload(t, user.uid, docId));
        cloudTrainingDocIdByLogicalId.set(t.id, docId);
      } catch (err) {
        handleFirestoreError(err, OperationType.CREATE, path);
      }
    }
  },

  /**
   * Real-time Firestore listener for signed-in user.
   * Performs non-destructive union merge: pulls cloud records down AND pushes missing local records up.
   */
  subscribeCloudCollections(
    user: User,
    callbacks: {
      onRoundsSynced: (mergedRounds: Round[]) => void;
      onTrainingSynced: (mergedTraining: AiTrainingExample[]) => void;
    }
  ): () => void {
    const uploadedRoundIds = new Set<string>();
    const uploadedTrainingIds = new Set<string>();

    const roundsQuery = query(collection(db, 'rounds'), where('ownerId', '==', user.uid));
    const unsubRounds = onSnapshot(
      roundsQuery,
      async (snapshot) => {
        const deletedIds = new Set(StorageService.getDeletedRoundIds());
        const cloudRounds: Round[] = [];
        const cloudLogicalIds = new Set<string>();

        snapshot.docs.forEach((d) => {
          const mapped = mapFirestoreDocToRound(d.data(), user.uid);
          if (mapped) {
            cloudLogicalIds.add(mapped.id);
            cloudLogicalIds.add(toSafeFirestoreId(mapped.id, 'round'));
            if (!deletedIds.has(mapped.id)) {
              cloudRounds.push(mapped);
            }
          }
        });

        // Clean up any tombstoned items on Firestore
        for (const delId of deletedIds) {
          const trackedDoc = cloudRoundDocIdByLogicalId.get(delId);
          if (trackedDoc && cloudLogicalIds.has(delId)) {
            deleteDoc(doc(db, 'rounds', trackedDoc)).catch(() => {});
          }
        }

        const localRounds = StorageService.getRounds();
        const merged = StorageService.mergeRounds(localRounds, cloudRounds);
        StorageService.saveRounds(merged);
        callbacks.onRoundsSynced(merged);
        void this.pushRoundsToServer(merged, Array.from(deletedIds));

        // Upload any local-only rounds to Firestore using UID-scoped document ID
        if (user.emailVerified) {
          for (const localItem of merged) {
            const safeBase = toSafeFirestoreId(localItem.id, 'round');
            if (
              !cloudLogicalIds.has(localItem.id) &&
              !cloudLogicalIds.has(safeBase) &&
              !uploadedRoundIds.has(safeBase)
            ) {
              uploadedRoundIds.add(safeBase);
              const docId = toUserScopedDocId(localItem.id, user.uid, 'round');
              const path = `rounds/${docId}`;
              try {
                await setDoc(
                  doc(db, 'rounds', docId),
                  buildFirestoreRoundPayload(localItem, user.uid, docId)
                );
                cloudRoundDocIdByLogicalId.set(localItem.id, docId);
              } catch (err) {
                handleFirestoreError(err, OperationType.CREATE, path);
              }
            }
          }
        }
      },
      (error) => {
        handleFirestoreError(error, OperationType.LIST, 'rounds');
      }
    );

    const trainingQuery = query(collection(db, 'aiTraining'), where('ownerId', '==', user.uid));
    const unsubTraining = onSnapshot(
      trainingQuery,
      async (snapshot) => {
        const deletedIds = new Set(StorageService.getDeletedTrainingIds());
        const cloudTraining: AiTrainingExample[] = [];
        const cloudLogicalIds = new Set<string>();

        snapshot.docs.forEach((d) => {
          const mapped = mapFirestoreDocToTraining(d.data(), user.uid);
          if (mapped) {
            cloudLogicalIds.add(mapped.id);
            cloudLogicalIds.add(toSafeFirestoreId(mapped.id, 'train'));
            if (!deletedIds.has(mapped.id)) {
              cloudTraining.push(mapped);
            }
          }
        });

        for (const delId of deletedIds) {
          const trackedDoc = cloudTrainingDocIdByLogicalId.get(delId);
          if (trackedDoc && cloudLogicalIds.has(delId)) {
            deleteDoc(doc(db, 'aiTraining', trackedDoc)).catch(() => {});
          }
        }

        const localTraining = StorageService.getAiTraining();
        const merged = StorageService.mergeAiTraining(localTraining, cloudTraining);
        StorageService.saveAiTraining(merged);
        callbacks.onTrainingSynced(merged);
        void this.pushTrainingToServer(merged, Array.from(deletedIds));

        if (user.emailVerified) {
          for (const localItem of merged) {
            const safeBase = toSafeFirestoreId(localItem.id, 'train');
            if (
              !cloudLogicalIds.has(localItem.id) &&
              !cloudLogicalIds.has(safeBase) &&
              !uploadedTrainingIds.has(safeBase)
            ) {
              uploadedTrainingIds.add(safeBase);
              const docId = toUserScopedDocId(localItem.id, user.uid, 'train');
              const path = `aiTraining/${docId}`;
              try {
                await setDoc(
                  doc(db, 'aiTraining', docId),
                  buildFirestoreTrainingPayload(localItem, user.uid, docId)
                );
                cloudTrainingDocIdByLogicalId.set(localItem.id, docId);
              } catch (err) {
                handleFirestoreError(err, OperationType.CREATE, path);
              }
            }
          }
        }
      },
      (error) => {
        handleFirestoreError(error, OperationType.LIST, 'aiTraining');
      }
    );

    return () => {
      unsubRounds();
      unsubTraining();
    };
  },
};
