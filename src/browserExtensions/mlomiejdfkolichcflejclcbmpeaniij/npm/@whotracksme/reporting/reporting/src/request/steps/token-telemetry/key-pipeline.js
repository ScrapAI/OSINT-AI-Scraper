import CachedEntryPipeline from './cached-entry-pipeline.js';
import logger from '../../../logger.js';

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


function getSiteTokensMap(siteTokens, key) {
  let siteTokensMap = siteTokens[key];
  if (!siteTokensMap) {
    siteTokensMap = {};
    siteTokens[key] = siteTokensMap;
  }
  return siteTokensMap;
}

class KeyPipeline extends CachedEntryPipeline {
  constructor(options) {
    super(options);
    this.primaryKey = 'hash';
  }

  newEntry() {
    return {
      created: Date.now(),
      dirty: true,
      sitesTokens: {},
      count: 0,
    };
  }

  updateCache({ hash, lastSent, key, tracker, created, sitesTokens, count }) {
    const stats = this.getFromCache(hash);
    if (stats.lastSent === undefined || lastSent > stats.lastSent) {
      stats.lastSent = lastSent;
    }
    stats.key = key;
    stats.tracker = tracker;
    stats.created = Math.min(stats.created, created);
    Object.keys(sitesTokens).forEach((site) => {
      const tokenMap = sitesTokens[site];
      const st = getSiteTokensMap(stats.sitesTokens, site);
      Object.entries(tokenMap).forEach(([token, safe]) => {
        st[token] = safe;
      });
    });
    stats.count = Math.max(stats.count, count);
  }

  serialiseEntry(hash, stats) {
    const { created, lastSent, key, tracker, sitesTokens, count } = stats;
    return {
      hash,
      key,
      tracker,
      created,
      lastSent: lastSent || '',
      sitesTokens,
      count,
    };
  }

  createMessagePayloads(toBeSent, batchLimit) {
    // grouping of key messages per site, up to batchLimit
    const groupedMessages = new Map();
    const overflow = [];
    toBeSent.forEach((tuple) => {
      const [, stats] = tuple;
      if (groupedMessages.size >= batchLimit) {
        overflow.push(tuple);
      } else {
        Object.entries(stats.sitesTokens).forEach(([site, tokens]) => {
          // if there are unsafe tokens in the group, make sure this entry is not grouped
          const unsafe = Object.values(tokens).some((t) => t === false);
          const extraKey = unsafe ? `${stats.tracker}:${stats.key}` : '';
          let entry = groupedMessages.get(`${site}${extraKey}`);
          if (!entry) {
            entry = [];
            groupedMessages.set(`${site}${extraKey}`, entry);
          }
          let tokensToSend = Object.entries(tokens);

          if (tokensToSend.length > this.options.KEY_TOKENS_LIMIT) {
            logger.warn(
              '[Request keys-pipeline]',
              `too many tokens for site="${site}" key="${stats.key}" tracker=${stats.tracker}`,
              `picking a random sample ${this.options.KEY_TOKENS_LIMIT} of ${tokensToSend.length}`,
            );
            tokensToSend = takeRandomSample(
              Object.entries(tokens),
              this.options.KEY_TOKENS_LIMIT,
            );
          }

          entry.push({
            ts: this.trustedClock.getTimeAsYYYYMMDD(),
            tracker: stats.tracker,
            key: stats.key,
            site,
            tokens: tokensToSend,
          });
        });
        Object.keys(stats.sitesTokens).forEach(
          (key) => delete stats.sitesTokens[key],
        );
        stats.count = 0;
      }
    });
    return {
      messages: [...groupedMessages.values()],
      overflow,
    };
  }

  hasData(entry) {
    return Object.keys(entry.sitesTokens).length > 0;
  }
}

function takeRandomSample(array, size) {
  const shuffled = [...array];
  let currentIndex = shuffled.length;

  // Fisher-Yates Shuffle Algorithm
  while (currentIndex !== 0) {
    const randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    const temporaryValue = shuffled[currentIndex];
    shuffled[currentIndex] = shuffled[randomIndex];
    shuffled[randomIndex] = temporaryValue;
  }

  return shuffled.slice(0, size);
}

export { KeyPipeline as default, getSiteTokensMap };
