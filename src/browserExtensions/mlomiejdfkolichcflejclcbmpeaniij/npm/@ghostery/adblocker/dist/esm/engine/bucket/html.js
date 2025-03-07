import NetworkFilter from '../../filters/network.js';
import CosmeticFilter from '../../filters/cosmetic.js';
import { optimizeNetwork, noopOptimizeNetwork, noopOptimizeCosmetic } from '../optimizer.js';
import ReverseIndex from '../reverse-index.js';
import { createLookupTokens } from './cosmetic.js';

class HTMLBucket {
    static deserialize(buffer, config) {
        const bucket = new HTMLBucket({ config });
        bucket.networkIndex = ReverseIndex.deserialize(buffer, NetworkFilter.deserialize, config.enableOptimizations ? optimizeNetwork : noopOptimizeNetwork, config);
        bucket.exceptionsIndex = ReverseIndex.deserialize(buffer, NetworkFilter.deserialize, config.enableOptimizations ? optimizeNetwork : noopOptimizeNetwork, config);
        bucket.cosmeticIndex = ReverseIndex.deserialize(buffer, CosmeticFilter.deserialize, noopOptimizeCosmetic, config);
        bucket.unhideIndex = ReverseIndex.deserialize(buffer, CosmeticFilter.deserialize, noopOptimizeCosmetic, config);
        return bucket;
    }
    constructor({ filters = [], config, }) {
        this.config = config;
        this.networkIndex = new ReverseIndex({
            config,
            deserialize: NetworkFilter.deserialize,
            filters: [],
            optimize: config.enableOptimizations ? optimizeNetwork : noopOptimizeNetwork,
        });
        this.exceptionsIndex = new ReverseIndex({
            config,
            deserialize: NetworkFilter.deserialize,
            filters: [],
            optimize: config.enableOptimizations ? optimizeNetwork : noopOptimizeNetwork,
        });
        this.cosmeticIndex = new ReverseIndex({
            config,
            deserialize: CosmeticFilter.deserialize,
            filters: [],
            optimize: noopOptimizeCosmetic,
        });
        this.unhideIndex = new ReverseIndex({
            config,
            deserialize: CosmeticFilter.deserialize,
            filters: [],
            optimize: noopOptimizeCosmetic,
        });
        if (filters.length !== 0) {
            this.update(filters, undefined);
        }
    }
    update(newFilters, removedFilters) {
        const networkFilters = [];
        const exceptionFilters = [];
        const cosmeticFilters = [];
        const unhideFilters = [];
        for (const filter of newFilters) {
            if (filter.isNetworkFilter()) {
                if (filter.isException()) {
                    exceptionFilters.push(filter);
                }
                else {
                    networkFilters.push(filter);
                }
            }
            else if (filter.isCosmeticFilter()) {
                if (filter.isUnhide()) {
                    unhideFilters.push(filter);
                }
                else {
                    cosmeticFilters.push(filter);
                }
            }
        }
        this.networkIndex.update(networkFilters, removedFilters);
        this.exceptionsIndex.update(exceptionFilters, removedFilters);
        this.cosmeticIndex.update(cosmeticFilters, removedFilters);
        this.unhideIndex.update(unhideFilters, removedFilters);
    }
    serialize(buffer) {
        this.networkIndex.serialize(buffer);
        this.exceptionsIndex.serialize(buffer);
        this.cosmeticIndex.serialize(buffer);
        this.unhideIndex.serialize(buffer);
    }
    getSerializedSize() {
        return (this.networkIndex.getSerializedSize() +
            this.exceptionsIndex.getSerializedSize() +
            this.cosmeticIndex.getSerializedSize() +
            this.unhideIndex.getSerializedSize());
    }
    getHTMLFilters(request, isFilterExcluded) {
        const networkFilters = [];
        const cosmeticFilters = [];
        const exceptions = [];
        const unhides = [];
        if (this.config.loadNetworkFilters === true) {
            this.networkIndex.iterMatchingFilters(request.getTokens(), (filter) => {
                if (filter.match(request) && !(isFilterExcluded === null || isFilterExcluded === void 0 ? void 0 : isFilterExcluded(filter))) {
                    networkFilters.push(filter);
                }
                return true;
            });
        }
        // If we found at least one candidate, check if we have exceptions.
        if (networkFilters.length !== 0) {
            this.exceptionsIndex.iterMatchingFilters(request.getTokens(), (filter) => {
                if (filter.match(request) && !(isFilterExcluded === null || isFilterExcluded === void 0 ? void 0 : isFilterExcluded(filter))) {
                    exceptions.push(filter);
                }
                return true;
            });
        }
        if (this.config.loadCosmeticFilters === true && request.isMainFrame()) {
            const { hostname, domain = '' } = request;
            const hostnameTokens = createLookupTokens(hostname, domain);
            this.cosmeticIndex.iterMatchingFilters(hostnameTokens, (filter) => {
                if (filter.match(hostname, domain) && !(isFilterExcluded === null || isFilterExcluded === void 0 ? void 0 : isFilterExcluded(filter))) {
                    cosmeticFilters.push(filter);
                }
                return true;
            });
            // If we found at least one candidate, check if we have unhidden rules.
            if (cosmeticFilters.length !== 0) {
                this.unhideIndex.iterMatchingFilters(hostnameTokens, (rule) => {
                    if (rule.match(hostname, domain) && !(isFilterExcluded === null || isFilterExcluded === void 0 ? void 0 : isFilterExcluded(rule))) {
                        unhides.push(rule);
                    }
                    return true;
                });
            }
        }
        return {
            networkFilters,
            cosmeticFilters,
            unhides,
            exceptions,
        };
    }
    getFilters() {
        const filters = [];
        return filters.concat(this.networkIndex.getFilters(), this.exceptionsIndex.getFilters(), this.cosmeticIndex.getFilters(), this.unhideIndex.getFilters());
    }
}

export { HTMLBucket as default };
