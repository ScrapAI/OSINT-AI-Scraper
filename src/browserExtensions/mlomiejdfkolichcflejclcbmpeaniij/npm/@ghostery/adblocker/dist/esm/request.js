import getRequestType from '../../../../@remusao/guess-url-type/dist/esm/index.js';
import { parse } from '../../../../tldts-experimental/dist/es6/index.js';
import { EMPTY_UINT32_ARRAY } from './data-view.js';
import { TOKENS_BUFFER } from './tokens-buffer.js';
import { fastHash, tokenizeNoSkipInPlace, HASH_INTERNAL_MULT, HASH_SEED } from './utils.js';

/*!
 * Copyright (c) 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
const TLDTS_OPTIONS = {
    extractHostname: true,
    mixedInputs: false,
    validateHostname: false,
};
const NORMALIZED_TYPE_TOKEN = {
    beacon: fastHash('type:beacon'),
    cspReport: fastHash('type:csp'),
    csp_report: fastHash('type:csp'),
    cspviolationreport: fastHash('type:cspviolationreport'),
    document: fastHash('type:document'),
    eventsource: fastHash('type:other'),
    fetch: fastHash('type:xhr'),
    font: fastHash('type:font'),
    image: fastHash('type:image'),
    imageset: fastHash('type:image'),
    mainFrame: fastHash('type:document'),
    main_frame: fastHash('type:document'),
    manifest: fastHash('type:other'),
    media: fastHash('type:media'),
    object: fastHash('type:object'),
    object_subrequest: fastHash('type:object'),
    other: fastHash('type:other'),
    ping: fastHash('type:ping'),
    prefetch: fastHash('type:other'),
    preflight: fastHash('type:preflight'),
    script: fastHash('type:script'),
    signedexchange: fastHash('type:signedexchange'),
    speculative: fastHash('type:other'),
    stylesheet: fastHash('type:stylesheet'),
    subFrame: fastHash('type:subdocument'),
    sub_frame: fastHash('type:subdocument'),
    texttrack: fastHash('type:other'),
    webSocket: fastHash('type:websocket'),
    web_manifest: fastHash('type:other'),
    websocket: fastHash('type:websocket'),
    xhr: fastHash('type:xhr'),
    xml_dtd: fastHash('type:other'),
    xmlhttprequest: fastHash('type:xhr'),
    xslt: fastHash('type:other'),
};
function hashHostnameBackward(hostname) {
    let hash = HASH_SEED;
    for (let j = hostname.length - 1; j >= 0; j -= 1) {
        hash = (hash * HASH_INTERNAL_MULT) ^ hostname.charCodeAt(j);
    }
    return hash >>> 0;
}
function getHashesFromLabelsBackward(hostname, end, startOfDomain) {
    TOKENS_BUFFER.reset();
    let hash = HASH_SEED;
    // Compute hash backward, label per label
    for (let i = end - 1; i >= 0; i -= 1) {
        const code = hostname.charCodeAt(i);
        // Process label
        if (code === 46 /* '.' */ && i < startOfDomain) {
            TOKENS_BUFFER.push(hash >>> 0);
        }
        // Update hash
        hash = (hash * HASH_INTERNAL_MULT) ^ code;
    }
    TOKENS_BUFFER.push(hash >>> 0);
    return TOKENS_BUFFER.slice();
}
/**
 * Given a hostname and its domain, return the hostname without the public
 * suffix. We know that the domain, with one less label on the left, will be a
 * the public suffix; and from there we know which trailing portion of
 * `hostname` we should remove.
 */
function getHostnameWithoutPublicSuffix(hostname, domain) {
    let hostnameWithoutPublicSuffix = null;
    const indexOfDot = domain.indexOf('.');
    if (indexOfDot !== -1) {
        const publicSuffix = domain.slice(indexOfDot + 1);
        hostnameWithoutPublicSuffix = hostname.slice(0, -publicSuffix.length - 1);
    }
    return hostnameWithoutPublicSuffix;
}
function getEntityHashesFromLabelsBackward(hostname, domain) {
    const hostnameWithoutPublicSuffix = getHostnameWithoutPublicSuffix(hostname, domain);
    if (hostnameWithoutPublicSuffix !== null) {
        return getHashesFromLabelsBackward(hostnameWithoutPublicSuffix, hostnameWithoutPublicSuffix.length, hostnameWithoutPublicSuffix.length);
    }
    return EMPTY_UINT32_ARRAY;
}
function getHostnameHashesFromLabelsBackward(hostname, domain) {
    return getHashesFromLabelsBackward(hostname, hostname.length, hostname.length - domain.length);
}
function isThirdParty(hostname, domain, sourceHostname, sourceDomain, type) {
    if (type === 'main_frame' || type === 'mainFrame') {
        return false;
    }
    else if (domain.length !== 0 && sourceDomain.length !== 0) {
        return domain !== sourceDomain;
    }
    else if (domain.length !== 0 && sourceHostname.length !== 0) {
        return domain !== sourceHostname;
    }
    else if (sourceDomain.length !== 0 && hostname.length !== 0) {
        return hostname !== sourceDomain;
    }
    return false;
}
class Request {
    /**
     * Create an instance of `Request` from raw request details.
     */
    static fromRawDetails({ requestId = '0', tabId = 0, url = '', hostname, domain, sourceUrl = '', sourceHostname, sourceDomain, type = 'main_frame', _originalRequestDetails, }) {
        url = url.toLowerCase();
        if (hostname === undefined || domain === undefined) {
            const parsed = parse(url, TLDTS_OPTIONS);
            hostname = hostname || parsed.hostname || '';
            domain = domain || parsed.domain || '';
        }
        // Initialize source URL
        if (sourceHostname === undefined || sourceDomain === undefined) {
            const parsed = parse(sourceHostname || sourceDomain || sourceUrl, TLDTS_OPTIONS);
            sourceHostname = sourceHostname || parsed.hostname || '';
            sourceDomain = sourceDomain || parsed.domain || sourceHostname || '';
        }
        return new Request({
            requestId,
            tabId,
            domain,
            hostname,
            url,
            sourceDomain,
            sourceHostname,
            sourceUrl,
            type,
            _originalRequestDetails,
        });
    }
    constructor({ requestId, tabId, type, domain, hostname, url, sourceDomain, sourceHostname, _originalRequestDetails, }) {
        // Lazy attributes
        this.tokens = undefined;
        this.hostnameHashes = undefined;
        this.entityHashes = undefined;
        this._originalRequestDetails = _originalRequestDetails;
        this.id = requestId;
        this.tabId = tabId;
        this.type = type;
        this.url = url;
        this.hostname = hostname;
        this.domain = domain;
        this.sourceHostnameHashes =
            sourceHostname.length === 0
                ? EMPTY_UINT32_ARRAY
                : getHostnameHashesFromLabelsBackward(sourceHostname, sourceDomain);
        this.sourceEntityHashes =
            sourceHostname.length === 0
                ? EMPTY_UINT32_ARRAY
                : getEntityHashesFromLabelsBackward(sourceHostname, sourceDomain);
        // Decide on partiness
        this.isThirdParty = isThirdParty(hostname, domain, sourceHostname, sourceDomain, type);
        this.isFirstParty = !this.isThirdParty;
        // Check protocol
        this.isSupported = true;
        if (this.type === 'websocket' || this.url.startsWith('ws:') || this.url.startsWith('wss:')) {
            this.isHttp = false;
            this.isHttps = false;
            this.type = 'websocket';
            this.isSupported = true;
        }
        else if (this.url.startsWith('http:')) {
            this.isHttp = true;
            this.isHttps = false;
        }
        else if (this.url.startsWith('https:')) {
            this.isHttps = true;
            this.isHttp = false;
        }
        else if (this.url.startsWith('data:')) {
            this.isHttp = false;
            this.isHttps = false;
            // Only keep prefix of URL
            const indexOfComa = this.url.indexOf(',');
            if (indexOfComa !== -1) {
                this.url = this.url.slice(0, indexOfComa);
            }
        }
        else {
            this.isHttp = false;
            this.isHttps = false;
            this.isSupported = false;
        }
    }
    getHostnameHashes() {
        if (this.hostnameHashes === undefined) {
            this.hostnameHashes =
                this.hostname.length === 0
                    ? EMPTY_UINT32_ARRAY
                    : getHostnameHashesFromLabelsBackward(this.hostname, this.domain);
        }
        return this.hostnameHashes;
    }
    getEntityHashes() {
        if (this.entityHashes === undefined) {
            this.entityHashes =
                this.hostname.length === 0
                    ? EMPTY_UINT32_ARRAY
                    : getEntityHashesFromLabelsBackward(this.hostname, this.domain);
        }
        return this.entityHashes;
    }
    getTokens() {
        if (this.tokens === undefined) {
            TOKENS_BUFFER.reset();
            for (const hash of this.sourceHostnameHashes) {
                TOKENS_BUFFER.push(hash);
            }
            // Add token corresponding to request type
            TOKENS_BUFFER.push(NORMALIZED_TYPE_TOKEN[this.type]);
            tokenizeNoSkipInPlace(this.url, TOKENS_BUFFER);
            this.tokens = TOKENS_BUFFER.slice();
        }
        return this.tokens;
    }
    isMainFrame() {
        return this.type === 'main_frame' || this.type === 'mainFrame';
    }
    isSubFrame() {
        return this.type === 'sub_frame' || this.type === 'subFrame';
    }
    /**
     * Calling this method will attempt to guess the type of a request based on
     * information found in `url` only. This can be useful to try and fine-tune
     * the type of a Request when it is not otherwise available or if it was
     * inferred as 'other'.
     */
    guessTypeOfRequest() {
        const currentType = this.type;
        this.type = getRequestType(this.url);
        if (currentType !== this.type) {
            this.tokens = undefined;
        }
        return this.type;
    }
}

export { NORMALIZED_TYPE_TOKEN, Request as default, getEntityHashesFromLabelsBackward, getHashesFromLabelsBackward, getHostnameHashesFromLabelsBackward, getHostnameWithoutPublicSuffix, hashHostnameBackward };
