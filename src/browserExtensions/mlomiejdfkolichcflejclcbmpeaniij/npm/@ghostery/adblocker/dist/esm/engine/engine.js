import Config from '../config.js';
import { StaticDataView, sizeOfByte, sizeOfASCII, sizeOfBool } from '../data-view.js';
import { EventEmitter } from '../events.js';
import { fetchLists, fetchResources, adsLists, adsAndTrackingLists, fullLists } from '../fetch.js';
import { normalizeSelector } from '../filters/cosmetic.js';
import { block } from '../filters/dsl.js';
import { parseFilters, FilterType } from '../lists.js';
import Request from '../request.js';
import Resources from '../resources.js';
import CosmeticFilterBucket, { createStylesheet } from './bucket/cosmetic.js';
import NetworkFilterBucket from './bucket/network.js';
import HTMLBucket from './bucket/html.js';
import { Metadata } from './metadata.js';
import Preprocessor, { Env } from '../preprocessor.js';
import PreprocessorBucket from './bucket/preprocessor.js';

/*!
 * Copyright (c) 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
const ENGINE_VERSION = 700;
function shouldApplyHideException(filters) {
    if (filters.length === 0) {
        return false;
    }
    // Get $Xhide filter with highest priority:
    // $Xhide,important > $Xhide > @@$Xhide
    let genericHideFilter;
    let currentScore = 0;
    for (const filter of filters) {
        // To encode priority between filters, we create a bitmask with the following:
        // $important,Xhide = 100 (takes precedence)
        // $Xhide           = 010 (exception to @@$Xhide)
        // @@$Xhide         = 001 (forbids Xhide filters)
        const score = (filter.isImportant() ? 4 : 0) | (filter.isException() ? 1 : 2);
        // Highest `score` has precedence
        if (score >= currentScore) {
            currentScore = score;
            genericHideFilter = filter;
        }
    }
    if (genericHideFilter === undefined) {
        return false;
    }
    // Check that there is at least one $generichide match and no exception
    return genericHideFilter.isException();
}
class FilterEngine extends EventEmitter {
    static fromCached(init, caching) {
        if (caching === undefined) {
            return init();
        }
        const { path, read, write } = caching;
        return read(path)
            .then((buffer) => this.deserialize(buffer))
            .catch(() => init().then((engine) => write(path, engine.serialize()).then(() => engine)));
    }
    static empty(config = {}) {
        return new this({ config });
    }
    /**
     * Create an instance of `FiltersEngine` (or subclass like `ElectronBlocker`,
     * etc.), from the list of subscriptions provided as argument (e.g.:
     * EasyList).
     *
     * Lists are fetched using the instance of `fetch` provided as a first
     * argument. Optionally resources.txt and config can be provided.
     */
    static fromLists(fetch, urls, config = {}, caching) {
        return this.fromCached(() => {
            const listsPromises = fetchLists(fetch, urls);
            const resourcesPromise = fetchResources(fetch);
            return Promise.all([listsPromises, resourcesPromise]).then(([lists, resources]) => {
                const engine = this.parse(lists.join('\n'), config);
                if (resources !== undefined) {
                    engine.updateResources(resources, '' + resources.length);
                }
                return engine;
            });
        }, caching);
    }
    /**
     * Initialize blocker of *ads only*.
     *
     * Attempt to initialize a blocking engine using a pre-built version served
     * from Ghostery's CDN. If this fails (e.g.: if no pre-built engine is available
     * for this version of the library), then falls-back to using `fromLists(...)`
     * method with the same subscriptions.
     */
    static fromPrebuiltAdsOnly(fetchImpl = fetch, caching) {
        return this.fromLists(fetchImpl, adsLists, {}, caching);
    }
    /**
     * Same as `fromPrebuiltAdsOnly(...)` but also contains rules to block
     * tracking (i.e.: using extra lists such as EasyPrivacy and more).
     */
    static fromPrebuiltAdsAndTracking(fetchImpl = fetch, caching) {
        return this.fromLists(fetchImpl, adsAndTrackingLists, {}, caching);
    }
    /**
     * Same as `fromPrebuiltAdsAndTracking(...)` but also contains annoyances
     * rules to block things like cookie notices.
     */
    static fromPrebuiltFull(fetchImpl = fetch, caching) {
        return this.fromLists(fetchImpl, fullLists, {}, caching);
    }
    static fromTrackerDB(rawJsonDump, options = {}) {
        const config = new Config(options);
        const metadata = new Metadata(rawJsonDump);
        const filters = [];
        for (const pattern of metadata.getPatterns()) {
            filters.push(...pattern.filters);
        }
        const engine = this.parse(filters.join('\n'), config);
        engine.metadata = metadata;
        return engine;
    }
    /**
     * Merges compatible engines into one.
     *
     * This action references objects from the source engines, including
     * network filters, cosmetic filters, preprocessors, metadata, and lists.
     * These objects are not deep-copied, so modifying them directly can have
     * unintended side effects.
     * However, resources are deep-copied from the first engine.
     *
     * Optionally, you can specify a second parameter to skip merging specific resources.
     * If resource merging is skipped, the resulting engine will be assigned empty resources.
     */
    static merge(engines, { skipResources = false, } = {}) {
        if (!engines || engines.length < 2) {
            throw new Error('merging engines requires at least two engines');
        }
        const config = engines[0].config;
        const lists = new Map();
        const networkFilters = new Map();
        const cosmeticFilters = new Map();
        const preprocessors = [];
        const metadata = {
            organizations: {},
            categories: {},
            patterns: {},
        };
        const compatibleConfigKeys = [];
        const configKeysMustMatch = Object.keys(config).filter(function (key) {
            return (typeof config[key] === 'boolean' && !compatibleConfigKeys.includes(key));
        });
        for (const engine of engines) {
            // Validate the config
            for (const configKey of configKeysMustMatch) {
                if (config[configKey] !== engine.config[configKey]) {
                    throw new Error(`config "${configKey}" of all merged engines must be the same`);
                }
            }
            const filters = engine.getFilters();
            for (const networkFilter of filters.networkFilters) {
                networkFilters.set(networkFilter.getId(), networkFilter);
            }
            for (const cosmeticFilter of filters.cosmeticFilters) {
                cosmeticFilters.set(cosmeticFilter.getId(), cosmeticFilter);
            }
            for (const preprocessor of engine.preprocessors.preprocessors) {
                preprocessors.push(preprocessor);
            }
            for (const [key, value] of engine.lists) {
                if (lists.has(key)) {
                    continue;
                }
                lists.set(key, value);
            }
            if (engine.metadata !== undefined) {
                for (const organization of engine.metadata.organizations.getValues()) {
                    if (metadata.organizations[organization.key] === undefined) {
                        metadata.organizations[organization.key] = organization;
                    }
                }
                for (const category of engine.metadata.categories.getValues()) {
                    if (metadata.categories[category.key] === undefined) {
                        metadata.categories[category.key] = category;
                    }
                }
                for (const pattern of engine.metadata.patterns.getValues()) {
                    if (metadata.patterns[pattern.key] === undefined) {
                        metadata.patterns[pattern.key] = pattern;
                    }
                }
            }
        }
        const engine = new this({
            networkFilters: Array.from(networkFilters.values()),
            cosmeticFilters: Array.from(cosmeticFilters.values()),
            preprocessors,
            lists,
            config,
        });
        if (Object.keys(metadata.categories).length +
            Object.keys(metadata.organizations).length +
            Object.keys(metadata.patterns).length !==
            0) {
            engine.metadata = new Metadata(metadata);
        }
        if (skipResources !== true) {
            for (const engine of engines.slice(1)) {
                if (engine.resources.checksum !== engines[0].resources.checksum) {
                    throw new Error(`resource checksum of all merged engines must match with the first one: "${engines[0].resources.checksum}" but got: "${engine.resources.checksum}"`);
                }
            }
            engine.resources = Resources.copy(engines[0].resources);
        }
        return engine;
    }
    static parse(filters, options = {}) {
        const config = new Config(options);
        return new this({
            ...parseFilters(filters, config),
            config,
        });
    }
    static deserialize(serialized) {
        const buffer = StaticDataView.fromUint8Array(serialized, {
            enableCompression: false,
        });
        // Before starting deserialization, we make sure that the version of the
        // serialized engine is the same as the current source code. If not, we
        // start fresh and create a new engine from the lists.
        const serializedEngineVersion = buffer.getUint16();
        if (ENGINE_VERSION !== serializedEngineVersion) {
            throw new Error(`serialized engine version mismatch, expected ${ENGINE_VERSION} but got ${serializedEngineVersion}`);
        }
        // Create a new engine with same options
        const config = Config.deserialize(buffer);
        // Optionally turn compression ON
        if (config.enableCompression) {
            buffer.enableCompression();
        }
        // Also make sure that the built-in checksum is correct. This allows to
        // detect data corruption and start fresh if the serialized version was
        // altered.
        if (config.integrityCheck) {
            const currentPos = buffer.pos;
            buffer.pos = serialized.length - 4;
            const checksum = buffer.checksum();
            const expected = buffer.getUint32();
            if (checksum !== expected) {
                throw new Error(`serialized engine checksum mismatch, expected ${expected} but got ${checksum}`);
            }
            buffer.pos = currentPos;
        }
        const engine = new this({ config });
        // Deserialize resources
        engine.resources = Resources.deserialize(buffer);
        // Deserialize lists
        const lists = new Map();
        const numberOfLists = buffer.getUint16();
        for (let i = 0; i < numberOfLists; i += 1) {
            lists.set(buffer.getASCII(), buffer.getASCII());
        }
        engine.lists = lists;
        // Deserialize preprocessors
        engine.preprocessors = PreprocessorBucket.deserialize(buffer);
        // Deserialize buckets
        engine.importants = NetworkFilterBucket.deserialize(buffer, config);
        engine.redirects = NetworkFilterBucket.deserialize(buffer, config);
        engine.filters = NetworkFilterBucket.deserialize(buffer, config);
        engine.exceptions = NetworkFilterBucket.deserialize(buffer, config);
        engine.csp = NetworkFilterBucket.deserialize(buffer, config);
        engine.cosmetics = CosmeticFilterBucket.deserialize(buffer, config);
        engine.hideExceptions = NetworkFilterBucket.deserialize(buffer, config);
        engine.htmlFilters = HTMLBucket.deserialize(buffer, config);
        // Optionally deserialize metadata
        const hasMetadata = buffer.getBool();
        if (hasMetadata) {
            engine.metadata = Metadata.deserialize(buffer);
        }
        buffer.seekZero();
        return engine;
    }
    constructor({ 
    // Optionally initialize the engine with filters
    cosmeticFilters = [], networkFilters = [], preprocessors = [], config = new Config(), lists = new Map(), } = {}) {
        super(); // init super-class EventEmitter
        this.config = new Config(config);
        // Subscription management: disabled by default
        this.lists = lists;
        // Preprocessors
        this.preprocessors = new PreprocessorBucket({});
        // $csp=
        this.csp = new NetworkFilterBucket({ config: this.config });
        // $elemhide
        // $generichide
        // $specifichide
        this.hideExceptions = new NetworkFilterBucket({ config: this.config });
        // @@filter
        this.exceptions = new NetworkFilterBucket({ config: this.config });
        // $important
        this.importants = new NetworkFilterBucket({ config: this.config });
        // $redirect
        this.redirects = new NetworkFilterBucket({ config: this.config });
        // All other filters
        this.filters = new NetworkFilterBucket({ config: this.config });
        // Cosmetic filters
        this.cosmetics = new CosmeticFilterBucket({ config: this.config });
        // HTML filters
        this.htmlFilters = new HTMLBucket({ config: this.config });
        // Injections
        this.resources = new Resources();
        if (networkFilters.length !== 0 || cosmeticFilters.length !== 0) {
            this.update({
                newCosmeticFilters: cosmeticFilters,
                newNetworkFilters: networkFilters,
                newPreprocessors: preprocessors,
            });
        }
    }
    isFilterExcluded(filter) {
        return this.preprocessors.isFilterExcluded(filter);
    }
    updateEnv(env) {
        this.preprocessors.updateEnv(env);
    }
    /**
     * Estimate the number of bytes needed to serialize this instance of
     * `FiltersEngine` using the `serialize(...)` method. It is used internally
     * by `serialize(...)` to allocate a buffer of the right size and you should
     * not have to call it yourself most of the time.
     *
     * There are cases where we cannot estimate statically the exact size of the
     * resulting buffer (due to alignement which needs to be performed); this
     * method will return a safe estimate which will always be at least equal to
     * the real number of bytes needed, or bigger (usually of a few bytes only:
     * ~20 bytes is to be expected).
     */
    getSerializedSize() {
        let estimatedSize = sizeOfByte() + // engine version
            this.config.getSerializedSize() +
            this.resources.getSerializedSize() +
            this.preprocessors.getSerializedSize() +
            this.filters.getSerializedSize() +
            this.exceptions.getSerializedSize() +
            this.importants.getSerializedSize() +
            this.redirects.getSerializedSize() +
            this.csp.getSerializedSize() +
            this.cosmetics.getSerializedSize() +
            this.hideExceptions.getSerializedSize() +
            this.htmlFilters.getSerializedSize() +
            4; // checksum
        // Estimate size of `this.lists` which stores information of checksum for each list.
        for (const [name, checksum] of this.lists) {
            estimatedSize += sizeOfASCII(name) + sizeOfASCII(checksum);
        }
        estimatedSize += sizeOfBool();
        if (this.metadata !== undefined) {
            estimatedSize += this.metadata.getSerializedSize();
        }
        return estimatedSize;
    }
    /**
     * Creates a binary representation of the full engine. It can be stored
     * on-disk for faster loading of the adblocker. The `deserialize` static
     * method of Engine can be used to restore the engine.
     */
    serialize(array) {
        const buffer = StaticDataView.fromUint8Array(array || new Uint8Array(this.getSerializedSize()), this.config);
        buffer.pushUint16(ENGINE_VERSION);
        // Config
        this.config.serialize(buffer);
        // Resources (js, resources)
        this.resources.serialize(buffer);
        // Serialize the state of lists (names and checksums)
        buffer.pushUint16(this.lists.size);
        for (const [name, value] of Array.from(this.lists.entries()).sort()) {
            buffer.pushASCII(name);
            buffer.pushASCII(value);
        }
        // Preprocessors
        this.preprocessors.serialize(buffer);
        // Filters buckets
        this.importants.serialize(buffer);
        this.redirects.serialize(buffer);
        this.filters.serialize(buffer);
        this.exceptions.serialize(buffer);
        this.csp.serialize(buffer);
        this.cosmetics.serialize(buffer);
        this.hideExceptions.serialize(buffer);
        this.htmlFilters.serialize(buffer);
        // Optionally serialize metadata
        buffer.pushBool(this.metadata !== undefined);
        if (this.metadata !== undefined) {
            this.metadata.serialize(buffer);
        }
        // Optionally append a checksum at the end
        if (this.config.integrityCheck) {
            buffer.pushUint32(buffer.checksum());
        }
        return buffer.subarray();
    }
    /**
     * Update engine with new filters or resources.
     */
    loadedLists() {
        return Array.from(this.lists.keys());
    }
    hasList(name, checksum) {
        return this.lists.has(name) && this.lists.get(name) === checksum;
    }
    /**
     * Update engine with `resources.txt` content.
     */
    updateResources(data, checksum) {
        if (this.resources.checksum === checksum) {
            return false;
        }
        this.resources = Resources.parse(data, { checksum });
        return true;
    }
    getFilters() {
        const cosmeticFilters = this.cosmetics.getFilters();
        const networkFilters = [
            ...this.filters.getFilters(),
            ...this.exceptions.getFilters(),
            ...this.importants.getFilters(),
            ...this.redirects.getFilters(),
            ...this.csp.getFilters(),
            ...this.hideExceptions.getFilters(),
        ];
        for (const filter of this.htmlFilters.getFilters()) {
            if (filter.isNetworkFilter()) {
                networkFilters.push(filter);
            }
            else if (filter.isCosmeticFilter()) {
                cosmeticFilters.push(filter);
            }
        }
        return {
            cosmeticFilters,
            networkFilters,
        };
    }
    /**
     * Update engine with new filters as well as optionally removed filters.
     */
    update({ newNetworkFilters = [], newCosmeticFilters = [], newPreprocessors = [], removedCosmeticFilters = [], removedNetworkFilters = [], removedPreprocessors = [], }, env = new Env()) {
        let updated = false;
        // Update preprocessors
        if (this.config.loadPreprocessors &&
            (newPreprocessors.length !== 0 || removedPreprocessors.length !== 0)) {
            updated = true;
            this.preprocessors.update({
                added: newPreprocessors,
                removed: removedPreprocessors,
            }, env);
        }
        const htmlFilters = [];
        // Update cosmetic filters
        if (this.config.loadCosmeticFilters &&
            (newCosmeticFilters.length !== 0 || removedCosmeticFilters.length !== 0)) {
            updated = true;
            const cosmeticFitlers = [];
            for (const filter of newCosmeticFilters) {
                if (filter.isHtmlFiltering()) {
                    htmlFilters.push(filter);
                }
                else {
                    cosmeticFitlers.push(filter);
                }
            }
            this.cosmetics.update(cosmeticFitlers, removedCosmeticFilters.length === 0 ? undefined : new Set(removedCosmeticFilters), this.config);
        }
        // Update network filters
        if (this.config.loadNetworkFilters &&
            (newNetworkFilters.length !== 0 || removedNetworkFilters.length !== 0)) {
            updated = true;
            const filters = [];
            const csp = [];
            const exceptions = [];
            const importants = [];
            const redirects = [];
            const hideExceptions = [];
            for (const filter of newNetworkFilters) {
                // NOTE: it's important to check for $generichide, $elemhide,
                // $specifichide and $csp before exceptions and important as we store
                // all of them in the same filter bucket. The check for exceptions is
                // done at match-time directly.
                if (filter.isCSP()) {
                    csp.push(filter);
                }
                else if (filter.isHtmlFilteringRule()) {
                    htmlFilters.push(filter);
                }
                else if (filter.isGenericHide() || filter.isSpecificHide()) {
                    hideExceptions.push(filter);
                }
                else if (filter.isException()) {
                    exceptions.push(filter);
                }
                else if (filter.isImportant()) {
                    importants.push(filter);
                }
                else if (filter.isRedirect()) {
                    redirects.push(filter);
                }
                else {
                    filters.push(filter);
                }
            }
            const removedNetworkFiltersSet = removedNetworkFilters.length === 0 ? undefined : new Set(removedNetworkFilters);
            // Update buckets in-place
            this.importants.update(importants, removedNetworkFiltersSet);
            this.redirects.update(redirects, removedNetworkFiltersSet);
            this.filters.update(filters, removedNetworkFiltersSet);
            if (this.config.loadExceptionFilters === true) {
                this.exceptions.update(exceptions, removedNetworkFiltersSet);
            }
            if (this.config.loadCSPFilters === true) {
                this.csp.update(csp, removedNetworkFiltersSet);
            }
            this.hideExceptions.update(hideExceptions, removedNetworkFiltersSet);
        }
        if (this.config.enableHtmlFiltering &&
            (htmlFilters.length !== 0 ||
                removedNetworkFilters.length !== 0 ||
                removedCosmeticFilters.length !== 0)) {
            const removeFilters = new Set([...removedNetworkFilters, ...removedCosmeticFilters]);
            this.htmlFilters.update(htmlFilters, removeFilters);
        }
        return updated;
    }
    updateFromDiff({ added, removed, preprocessors }, env) {
        const newCosmeticFilters = [];
        const newNetworkFilters = [];
        const newPreprocessors = [];
        const removedCosmeticFilters = [];
        const removedNetworkFilters = [];
        const removedPreprocessors = [];
        if (removed !== undefined && removed.length !== 0) {
            const { networkFilters, cosmeticFilters } = parseFilters(removed.join('\n'), this.config);
            Array.prototype.push.apply(removedCosmeticFilters, cosmeticFilters);
            Array.prototype.push.apply(removedNetworkFilters, networkFilters);
        }
        if (added !== undefined && added.length !== 0) {
            const { networkFilters, cosmeticFilters } = parseFilters(added.join('\n'), this.config);
            Array.prototype.push.apply(newCosmeticFilters, cosmeticFilters);
            Array.prototype.push.apply(newNetworkFilters, networkFilters);
        }
        if (preprocessors !== undefined) {
            for (const [condition, details] of Object.entries(preprocessors)) {
                if (details.removed !== undefined && details.removed.length !== 0) {
                    const { networkFilters, cosmeticFilters } = parseFilters(details.removed.join('\n'), this.config);
                    const filterIDs = new Set([]
                        .concat(cosmeticFilters.map((filter) => filter.getId()))
                        .concat(networkFilters.map((filter) => filter.getId())));
                    removedPreprocessors.push(new Preprocessor({
                        condition,
                        filterIDs,
                    }));
                }
                if (details.added !== undefined && details.added.length !== 0) {
                    const { networkFilters, cosmeticFilters } = parseFilters(details.added.join('\n'), this.config);
                    const filterIDs = new Set([]
                        .concat(cosmeticFilters.map((filter) => filter.getId()))
                        .concat(networkFilters.map((filter) => filter.getId())));
                    newPreprocessors.push(new Preprocessor({
                        condition,
                        filterIDs,
                    }));
                }
            }
        }
        return this.update({
            newCosmeticFilters,
            newNetworkFilters,
            newPreprocessors,
            removedCosmeticFilters: removedCosmeticFilters.map((f) => f.getId()),
            removedNetworkFilters: removedNetworkFilters.map((f) => f.getId()),
            removedPreprocessors,
        }, env);
    }
    /**
     * Return a list of HTML filtering rules.
     */
    getHtmlFilters(request) {
        const htmlSelectors = [];
        if (this.config.enableHtmlFiltering === false) {
            return htmlSelectors;
        }
        const { networkFilters, exceptions, cosmeticFilters, unhides } = this.htmlFilters.getHTMLFilters(request, this.isFilterExcluded.bind(this));
        if (cosmeticFilters.length !== 0) {
            const unhideMap = new Map(unhides.map((unhide) => [unhide.getSelector(), unhide]));
            for (const filter of cosmeticFilters) {
                const extended = filter.getExtendedSelector();
                if (extended === undefined) {
                    continue;
                }
                const unhide = unhideMap.get(filter.getSelector());
                if (unhide === undefined) {
                    htmlSelectors.push(extended);
                }
                this.emit('filter-matched', { filter, exception: unhide }, {
                    request,
                    filterType: FilterType.COSMETIC,
                });
            }
        }
        if (networkFilters.length !== 0) {
            const exceptionsMap = new Map();
            let replaceDisabledException;
            for (const exception of exceptions) {
                const optionValue = exception.optionValue;
                if (optionValue === '') {
                    replaceDisabledException = exception;
                    break;
                }
                exceptionsMap.set(optionValue, exception);
            }
            for (const filter of networkFilters) {
                const modifier = filter.getHtmlModifier();
                if (modifier === null) {
                    continue;
                }
                const exception = replaceDisabledException || exceptionsMap.get(filter.optionValue);
                this.emit('filter-matched', { filter, exception }, {
                    request,
                    filterType: FilterType.NETWORK,
                });
                if (exception === undefined) {
                    htmlSelectors.push(['replace', modifier]);
                }
            }
        }
        if (htmlSelectors.length !== 0) {
            this.emit('html-filtered', htmlSelectors, request.url);
        }
        return htmlSelectors;
    }
    /**
     * Given `hostname` and `domain` of a page (or frame), return the list of
     * styles and scripts to inject in the page.
     */
    getCosmeticsFilters({ 
    // Page information
    url, hostname, domain, 
    // DOM information
    classes, hrefs, ids, 
    // Allows to specify which rules to return
    getBaseRules = true, getInjectionRules = true, getExtendedRules = true, getRulesFromDOM = true, getRulesFromHostname = true, 
    // inject extended selector filters
    injectPureHasSafely = false, hidingStyle, callerContext, }) {
        if (this.config.loadCosmeticFilters === false) {
            return {
                active: false,
                extended: [],
                scripts: [],
                styles: '',
            };
        }
        domain || (domain = '');
        let allowGenericHides = true;
        let allowSpecificHides = true;
        const exceptions = this.hideExceptions.matchAll(Request.fromRawDetails({
            domain,
            hostname,
            url,
            sourceDomain: '',
            sourceHostname: '',
            sourceUrl: '',
        }), this.isFilterExcluded.bind(this));
        const genericHides = [];
        const specificHides = [];
        for (const filter of exceptions) {
            if (filter.isElemHide()) {
                allowGenericHides = false;
                allowSpecificHides = false;
                break;
            }
            if (filter.isSpecificHide()) {
                specificHides.push(filter);
            }
            else if (filter.isGenericHide()) {
                genericHides.push(filter);
            }
        }
        if (allowGenericHides === true) {
            allowGenericHides = shouldApplyHideException(genericHides) === false;
        }
        if (allowSpecificHides === true) {
            allowSpecificHides = shouldApplyHideException(specificHides) === false;
        }
        // Lookup injections as well as stylesheets
        const { filters, unhides } = this.cosmetics.getCosmeticsFilters({
            domain,
            hostname,
            classes,
            hrefs,
            ids,
            allowGenericHides,
            allowSpecificHides,
            getRulesFromDOM,
            getRulesFromHostname,
            hidingStyle,
            isFilterExcluded: this.isFilterExcluded.bind(this),
        });
        let injectionsDisabled = false;
        const unhideExceptions = new Map();
        for (const unhide of unhides) {
            if (unhide.isScriptInject() === true &&
                unhide.isUnhide() === true &&
                unhide.getSelector().length === 0) {
                injectionsDisabled = true;
            }
            unhideExceptions.set(normalizeSelector(unhide, this.resources.getScriptletCanonicalName.bind(this.resources)), unhide);
        }
        const injections = [];
        const styleFilters = [];
        const extendedFilters = [];
        const pureHasFilters = [];
        if (filters.length !== 0) {
            // Apply unhide rules + dispatch
            for (const filter of filters) {
                // Make sure `rule` is not un-hidden by a #@# filter
                const exception = unhideExceptions.get(normalizeSelector(filter, this.resources.getScriptletCanonicalName.bind(this.resources)));
                if (exception !== undefined) {
                    continue;
                }
                let applied = false;
                // Dispatch filters in `injections` or `styles` depending on type
                if (filter.isScriptInject() === true) {
                    if (getInjectionRules === true && injectionsDisabled === false) {
                        injections.push(filter);
                        applied = true;
                    }
                }
                else if (filter.isExtended()) {
                    if (injectPureHasSafely && filter.isPureHasSelector()) {
                        pureHasFilters.push(filter);
                        applied = true;
                    }
                    if (this.config.loadExtendedSelectors && getExtendedRules === true) {
                        extendedFilters.push(filter);
                        applied = true;
                    }
                }
                else {
                    styleFilters.push(filter);
                    applied = true;
                }
                if (applied) {
                    this.emit('filter-matched', {
                        filter,
                        exception,
                    }, {
                        url,
                        callerContext,
                        filterType: FilterType.COSMETIC,
                    });
                }
            }
        }
        // Perform interpolation for injected scripts
        const scripts = [];
        for (const injection of injections) {
            const script = injection.getScript(this.resources.getScriptlet.bind(this.resources));
            if (script !== undefined) {
                this.emit('script-injected', script, url);
                scripts.push(script);
            }
        }
        const stylesheets = this.cosmetics.getStylesheetsFromFilters({
            filters: styleFilters,
            extendedFilters,
        }, { getBaseRules, allowGenericHides, hidingStyle });
        const { extended } = stylesheets;
        let { stylesheet } = stylesheets;
        for (const safeHasFilter of pureHasFilters) {
            stylesheet += `\n\n${createStylesheet([safeHasFilter.getSelector()], hidingStyle)}`;
        }
        // Emit events
        if (stylesheet.length !== 0) {
            this.emit('style-injected', stylesheet, url);
        }
        return {
            active: true,
            extended,
            scripts,
            styles: stylesheet,
        };
    }
    /**
     * Given a `request`, return all matching network filters found in the engine.
     */
    matchAll(request) {
        const filters = [];
        if (request.isSupported) {
            Array.prototype.push.apply(filters, this.importants.matchAll(request, this.isFilterExcluded.bind(this)));
            Array.prototype.push.apply(filters, this.filters.matchAll(request, this.isFilterExcluded.bind(this)));
            Array.prototype.push.apply(filters, this.exceptions.matchAll(request, this.isFilterExcluded.bind(this)));
            Array.prototype.push.apply(filters, this.csp.matchAll(request, this.isFilterExcluded.bind(this)));
            Array.prototype.push.apply(filters, this.hideExceptions.matchAll(request, this.isFilterExcluded.bind(this)));
            Array.prototype.push.apply(filters, this.redirects.matchAll(request, this.isFilterExcluded.bind(this)));
        }
        return new Set(filters);
    }
    /**
     * Given a "main_frame" request, check if some content security policies
     * should be injected in the page.
     */
    getCSPDirectives(request) {
        if (!this.config.loadNetworkFilters) {
            return undefined;
        }
        if (request.isSupported !== true || request.isMainFrame() === false) {
            return undefined;
        }
        const matches = this.csp.matchAll(request, this.isFilterExcluded.bind(this));
        // No $csp filter found
        if (matches.length === 0) {
            return undefined;
        }
        // Collect all CSP directives and keep track of exceptions
        const cspExceptions = new Map();
        const cspFilters = [];
        for (const filter of matches) {
            if (filter.isException()) {
                if (filter.csp === undefined) {
                    // All CSP directives are disabled for this site
                    this.emit('filter-matched', { exception: filter }, { request, filterType: FilterType.NETWORK });
                    return undefined;
                }
                cspExceptions.set(filter.csp, filter);
            }
            else {
                cspFilters.push(filter);
            }
        }
        if (cspFilters.length === 0) {
            return undefined;
        }
        const enabledCsp = new Set();
        // Combine all CSPs (except the black-listed ones)
        for (const filter of cspFilters.values()) {
            const exception = cspExceptions.get(filter.csp);
            if (exception === undefined) {
                enabledCsp.add(filter.csp);
            }
            this.emit('filter-matched', { filter, exception }, { request, filterType: FilterType.NETWORK });
        }
        const csps = Array.from(enabledCsp).join('; ');
        if (csps.length > 0) {
            this.emit('csp-injected', request, csps);
        }
        return csps;
    }
    /**
     * Decide if a network request (usually from WebRequest API) should be
     * blocked, redirected or allowed.
     */
    match(request, withMetadata = false) {
        const result = {
            exception: undefined,
            filter: undefined,
            match: false,
            redirect: undefined,
            metadata: undefined,
        };
        if (!this.config.loadNetworkFilters) {
            return result;
        }
        if (request.isSupported) {
            // Check the filters in the following order:
            // 1. $important (not subject to exceptions)
            // 2. redirection ($redirect=resource)
            // 3. normal filters
            // 4. exceptions
            result.filter = this.importants.match(request, this.isFilterExcluded.bind(this));
            let redirectNone;
            let redirectRule;
            // If `result.filter` is `undefined`, it means there was no $important
            // filter found so far. We look for a $redirect filter.  There is some
            // extra logic to handle special cases like redirect-rule and
            // redirect=none.
            //
            // * If redirect=none is found, then cancel all redirects.
            // * Else if redirect-rule is found, only redirect if request would be blocked.
            // * Else if redirect is found, redirect.
            if (result.filter === undefined) {
                const redirects = this.redirects
                    .matchAll(request, this.isFilterExcluded.bind(this))
                    // highest priorty wins
                    .sort((a, b) => b.getRedirectPriority() - a.getRedirectPriority());
                if (redirects.length !== 0) {
                    for (const filter of redirects) {
                        if (filter.getRedirectResource() === 'none') {
                            redirectNone = filter;
                        }
                        else if (filter.isRedirectRule()) {
                            if (redirectRule === undefined) {
                                redirectRule = filter;
                            }
                        }
                        else if (result.filter === undefined) {
                            result.filter = filter;
                        }
                    }
                }
                // If `result.filter` is still `undefined`, it means that there was no
                // redirection rule triggered for the request. We look for a normal
                // match.
                if (result.filter === undefined) {
                    result.filter = this.filters.match(request, this.isFilterExcluded.bind(this));
                    // If we found a match, and a `$redirect-rule` as found previously,
                    // then we transform the match into a redirect, following the
                    // semantics of redirect-rule.
                    if (redirectRule !== undefined && result.filter !== undefined) {
                        result.filter = redirectRule;
                    }
                }
                // If we found either a redirection rule or a normal match, then check
                // for exceptions which could apply on the request and un-block it.
                if (result.filter !== undefined) {
                    result.exception = this.exceptions.match(request, this.isFilterExcluded.bind(this));
                }
            }
            // If there was a redirect match and no exception was found, then we
            // proceed and process the redirect rule. This means two things:
            //
            // 1. Check if a redirect=none rule was found, which acts as exception.
            // 2. If no exception was found, prepare `result.redirect` response.
            if (result.filter !== undefined &&
                result.exception === undefined &&
                result.filter.isRedirect()) {
                if (redirectNone !== undefined) {
                    result.exception = redirectNone;
                }
                else {
                    result.redirect = this.resources.getResource(result.filter.getRedirectResource());
                }
            }
        }
        result.match = result.exception === undefined && result.filter !== undefined;
        if (result.filter) {
            this.emit('filter-matched', { filter: result.filter, exception: result.exception }, { request, filterType: FilterType.NETWORK });
        }
        if (result.exception !== undefined) {
            this.emit('request-whitelisted', request, result);
        }
        else if (result.redirect !== undefined) {
            this.emit('request-redirected', request, result);
        }
        else if (result.filter !== undefined) {
            this.emit('request-blocked', request, result);
        }
        else {
            this.emit('request-allowed', request, result);
        }
        if (withMetadata === true && result.filter !== undefined && this.metadata) {
            result.metadata = this.metadata.fromFilter(result.filter);
        }
        return result;
    }
    getPatternMetadata(request, { getDomainMetadata = false } = {}) {
        if (this.metadata === undefined) {
            return [];
        }
        const seenPatterns = new Set();
        const patterns = [];
        for (const filter of this.matchAll(request)) {
            for (const patternInfo of this.metadata.fromFilter(filter)) {
                if (!seenPatterns.has(patternInfo.pattern.key)) {
                    seenPatterns.add(patternInfo.pattern.key);
                    patterns.push(patternInfo);
                }
            }
        }
        if (getDomainMetadata) {
            for (const patternInfo of this.metadata.fromDomain(request.hostname)) {
                if (!seenPatterns.has(patternInfo.pattern.key)) {
                    seenPatterns.add(patternInfo.pattern.key);
                    patterns.push(patternInfo);
                }
            }
        }
        return patterns;
    }
    blockScripts() {
        this.updateFromDiff({
            added: [block().scripts().redirectTo('javascript').toString()],
        });
        return this;
    }
    blockImages() {
        this.updateFromDiff({
            added: [block().images().redirectTo('png').toString()],
        });
        return this;
    }
    blockMedias() {
        this.updateFromDiff({
            added: [block().medias().redirectTo('mp4').toString()],
        });
        return this;
    }
    blockFrames() {
        this.updateFromDiff({
            added: [block().frames().redirectTo('html').toString()],
        });
        return this;
    }
    blockFonts() {
        this.updateFromDiff({
            added: [block().fonts().toString()],
        });
        return this;
    }
    blockStyles() {
        this.updateFromDiff({
            added: [block().styles().toString()],
        });
        return this;
    }
}

export { ENGINE_VERSION, FilterEngine as default };
