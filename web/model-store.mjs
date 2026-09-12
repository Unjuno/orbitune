const DB_NAME = 'orbitune-model-cache-v1';
const STORE_NAME = 'models';
const DB_VERSION = 1;

function normalizeSha256(value) {
  const normalized = String(value || '').trim().toLowerCase();
  if (!/^[0-9a-f]{64}$/.test(normalized)) throw new Error('model cache key must be a SHA-256 hex string');
  return normalized;
}

function cloneArrayBuffer(value) {
  if (value instanceof ArrayBuffer) return value.slice(0);
  if (ArrayBuffer.isView(value)) return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  throw new Error('model bytes must be an ArrayBuffer or typed array');
}

function requestResult(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('IndexedDB request failed'));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error || new Error('IndexedDB transaction failed'));
    transaction.onabort = () => reject(transaction.error || new Error('IndexedDB transaction aborted'));
  });
}

export class IndexedDbModelStore {
  constructor({ indexedDb = globalThis.indexedDB, dbName = DB_NAME, storeName = STORE_NAME } = {}) {
    this.indexedDb = indexedDb;
    this.dbName = dbName;
    this.storeName = storeName;
    this.dbPromise = null;
  }

  get available() { return Boolean(this.indexedDb?.open); }

  async open() {
    if (!this.available) return null;
    if (!this.dbPromise) {
      this.dbPromise = new Promise((resolve, reject) => {
        const request = this.indexedDb.open(this.dbName, DB_VERSION);
        request.onupgradeneeded = () => {
          const db = request.result;
          if (!db.objectStoreNames.contains(this.storeName)) db.createObjectStore(this.storeName, { keyPath: 'sha256' });
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error || new Error('Unable to open model cache'));
      });
    }
    return this.dbPromise;
  }

  async get(sha256) {
    const key = normalizeSha256(sha256);
    const db = await this.open();
    if (!db) return null;
    const transaction = db.transaction(this.storeName, 'readonly');
    const entry = await requestResult(transaction.objectStore(this.storeName).get(key));
    if (!entry?.bytes) return null;
    return cloneArrayBuffer(entry.bytes);
  }

  async put(sha256, bytes, metadata = {}) {
    const key = normalizeSha256(sha256);
    const db = await this.open();
    if (!db) return false;
    const transaction = db.transaction(this.storeName, 'readwrite');
    transaction.objectStore(this.storeName).put({
      sha256: key,
      bytes: cloneArrayBuffer(bytes),
      metadata: { ...metadata },
      stored_at: new Date().toISOString(),
    });
    await transactionDone(transaction);
    return true;
  }

  async delete(sha256) {
    const key = normalizeSha256(sha256);
    const db = await this.open();
    if (!db) return false;
    const transaction = db.transaction(this.storeName, 'readwrite');
    transaction.objectStore(this.storeName).delete(key);
    await transactionDone(transaction);
    return true;
  }
}
