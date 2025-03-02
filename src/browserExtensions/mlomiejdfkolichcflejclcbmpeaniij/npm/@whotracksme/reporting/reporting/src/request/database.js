import { openDB } from '../../../npm/idb/build/index.js';

/**
 * WhoTracks.Me
 * https://whotracks.me/
 *
 * Copyright 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0
 */


class IDBWrapper {
  constructor(db, tableName, primaryKey) {
    this.db = db;
    this.tableName = tableName;
    this.primaryKey = primaryKey;
  }

  async put(value) {
    await this.db.put(this.tableName, value);
  }

  async clear() {
    await this.db.clear(this.tableName);
  }

  async uniqueKeys() {
    return this.db.getAllKeys(this.tableName);
  }

  async count() {
    return this.db.count(this.tableName);
  }

  async bulkPut(rows) {
    if (rows.length === 0) {
      return;
    }
    const tx = this.db.transaction(this.tableName, 'readwrite');
    await Promise.all(rows.map((row) => tx.store.put(row)));
    await tx.done;
  }

  async bulkDelete(keys) {
    if (keys.length === 0) {
      return;
    }
    const tx = this.db.transaction(this.tableName, 'readwrite');
    await Promise.all(keys.map((key) => tx.store.delete(key)));
    await tx.done;
  }

  async where({ primaryKey }) {
    if (primaryKey === this.primaryKey) {
      return await this.db.getAll(this.tableName);
    } else {
      return await this.db.getAllFromIndex(this.tableName, primaryKey);
    }
  }
}

class AttrackDatabase {
  constructor() {
    this.tableName = 'antitracking';
    this.db = null;
    this._ready = null;
  }

  async init() {
    let resolver;
    this._ready = new Promise((resolve) => {
      resolver = resolve;
    });
    this.db = await openDB(this.tableName, 21, {
      async upgrade(db, oldVersion, newVersion) {
        if (oldVersion < 1) {
          const tokenBlockedStore = db.createObjectStore('tokenBlocked', {
            keyPath: 'token',
          });
          tokenBlockedStore.createIndex('token', 'token');
          tokenBlockedStore.createIndex('expires', 'expires');

          const tokensStore = db.createObjectStore('tokens', {
            keyPath: 'token',
          });
          tokensStore.createIndex('lastSent', 'lastSent');
          tokensStore.createIndex('created', 'created');

          const keysStore = db.createObjectStore('keys', { keyPath: 'hash' });
          keysStore.createIndex('lastSent', 'lastSent');
          keysStore.createIndex('created', 'created');
        }

        if (newVersion >= 21) {
          db.createObjectStore('keyval');
        }

        if (db.objectStoreNames.contains('tokenDomain')) {
          db.deleteObjectStore('tokenDomain');
        }

        if (db.objectStoreNames.contains('requestKeyValue')) {
          db.deleteObjectStore('requestKeyValue');
        }
      },
    });
    resolver();
  }

  unload() {
    if (this.db !== null) {
      this.db.close();
      this.db = null;
    }
  }

  get ready() {
    if (this._ready === null) {
      return Promise.reject(new Error('init not called'));
    }
    return this._ready;
  }

  get tokenBlocked() {
    return new IDBWrapper(this.db, 'tokenBlocked', 'token');
  }

  get tokens() {
    return new IDBWrapper(this.db, 'tokens', 'token');
  }

  get keys() {
    return new IDBWrapper(this.db, 'keys', 'hash');
  }

  async get(key) {
    await this._ready;
    return this.db.get('keyval', key);
  }

  async set(key, val) {
    await this._ready;
    return this.db.put('keyval', val, key);
  }
}

export { AttrackDatabase as default };
