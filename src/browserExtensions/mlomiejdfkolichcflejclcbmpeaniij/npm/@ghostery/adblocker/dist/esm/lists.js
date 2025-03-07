import Config from './config.js';
import CosmeticFilter from './filters/cosmetic.js';
import NetworkFilter from './filters/network.js';
import Preprocessor, { detectPreprocessor, PreprocessorTokens } from './preprocessor.js';
import { fastStartsWith, fastStartsWithFrom } from './utils.js';

/*!
 * Copyright (c) 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
var FilterType;
(function (FilterType) {
    FilterType[FilterType["NOT_SUPPORTED"] = 0] = "NOT_SUPPORTED";
    FilterType[FilterType["NETWORK"] = 1] = "NETWORK";
    FilterType[FilterType["COSMETIC"] = 2] = "COSMETIC";
    // available only with `extendedNonSupportedTypes` option for #detectFilterType
    FilterType[FilterType["NOT_SUPPORTED_EMPTY"] = 100] = "NOT_SUPPORTED_EMPTY";
    FilterType[FilterType["NOT_SUPPORTED_COMMENT"] = 101] = "NOT_SUPPORTED_COMMENT";
    FilterType[FilterType["NOT_SUPPORTED_ADGUARD"] = 102] = "NOT_SUPPORTED_ADGUARD";
})(FilterType || (FilterType = {}));
/**
 * Given a single line (string), checks if this would likely be a cosmetic
 * filter, a network filter or something that is not supported. This check is
 * performed before calling a more specific parser to create an instance of
 * `NetworkFilter` or `CosmeticFilter`.
 */
function detectFilterType(line, { extendedNonSupportedTypes = false } = {}) {
    // Ignore empty line
    if (line.length === 0 || line.length === 1) {
        if (extendedNonSupportedTypes) {
            return FilterType.NOT_SUPPORTED_EMPTY;
        }
        return FilterType.NOT_SUPPORTED;
    }
    // Ignore comments
    const firstCharCode = line.charCodeAt(0);
    const secondCharCode = line.charCodeAt(1);
    if (firstCharCode === 33 /* '!' */ ||
        (firstCharCode === 35 /* '#' */ && secondCharCode <= 32) ||
        (firstCharCode === 91 /* '[' */ && fastStartsWith(line, '[Adblock'))) {
        if (extendedNonSupportedTypes) {
            return FilterType.NOT_SUPPORTED_COMMENT;
        }
        return FilterType.NOT_SUPPORTED;
    }
    // Fast heuristics to detect network filters
    const lastCharCode = line.charCodeAt(line.length - 1);
    if ((firstCharCode === 36 /* '$' */ &&
        secondCharCode !== 36 &&
        secondCharCode !== 64) /* $$ and $@ as those may be Adguard HTML filtering rules */ ||
        firstCharCode === 38 /* '&' */ ||
        firstCharCode === 42 /* '*' */ ||
        firstCharCode === 45 /* '-' */ ||
        firstCharCode === 46 /* '.' */ ||
        firstCharCode === 47 /* '/' */ ||
        firstCharCode === 58 /* ':' */ ||
        firstCharCode === 61 /* '=' */ ||
        firstCharCode === 63 /* '?' */ ||
        firstCharCode === 64 /* '@' */ ||
        firstCharCode === 95 /* '_' */ ||
        firstCharCode === 124 /* '|' */ ||
        lastCharCode === 124 /* '|' */) {
        return FilterType.NETWORK;
    }
    // Ignore Adguard cosmetics
    // `$$` = HTML filtering rules
    const dollarIndex = line.indexOf('$');
    if (dollarIndex !== -1 && dollarIndex !== line.length - 1) {
        const afterDollarIndex = dollarIndex + 1;
        const afterDollarCharCode = line.charCodeAt(afterDollarIndex);
        // Ignore Adguard HTML rewrite rules
        if (afterDollarCharCode === 36 /* '$' */ ||
            (afterDollarCharCode === 64 /* '@' */ &&
                fastStartsWithFrom(line, /* $@$ */ '@$', afterDollarIndex))) {
            if (extendedNonSupportedTypes) {
                return FilterType.NOT_SUPPORTED_ADGUARD;
            }
            return FilterType.NOT_SUPPORTED;
        }
    }
    // Check if filter is cosmetics
    const sharpIndex = line.indexOf('#');
    if (sharpIndex !== -1 && sharpIndex !== line.length - 1) {
        const afterSharpIndex = sharpIndex + 1;
        const afterSharpCharCode = line.charCodeAt(afterSharpIndex);
        if (afterSharpCharCode === 35 /* '#'*/ ||
            (afterSharpCharCode === 64 /* '@' */ &&
                fastStartsWithFrom(line, /* #@# */ '@#', afterSharpIndex))
        // TODO - support ADB/AdGuard extended css selectors
        // || (afterSharpCharCode === 63 /* '?' */ &&
        //   fastStartsWithFrom(line, /* #?# */ '?#', afterSharpIndex))
        ) {
            // Parse supported cosmetic filter
            // `##` `#@#`
            return FilterType.COSMETIC;
        }
        else if ((afterSharpCharCode === 64 /* '@'*/ &&
            (fastStartsWithFrom(line, /* #@$# */ '@$#', afterSharpIndex) ||
                fastStartsWithFrom(line, /* #@%# */ '@%#', afterSharpIndex) ||
                fastStartsWithFrom(line, /* #@?# */ '@?#', afterSharpIndex))) ||
            (afterSharpCharCode === 37 /* '%' */ &&
                fastStartsWithFrom(line, /* #%# */ '%#', afterSharpIndex)) ||
            (afterSharpCharCode === 36 /* '$' */ &&
                (fastStartsWithFrom(line, /* #$# */ '$#', afterSharpIndex) ||
                    fastStartsWithFrom(line, /* #$?# */ '$?#', afterSharpIndex))) ||
            (afterSharpCharCode === 63 /* '?' */ &&
                fastStartsWithFrom(line, /* #?# */ '?#', afterSharpIndex))) {
            if (extendedNonSupportedTypes) {
                return FilterType.NOT_SUPPORTED_ADGUARD;
            }
            return FilterType.NOT_SUPPORTED;
        }
    }
    // Everything else is a network filter
    return FilterType.NETWORK;
}
function parseFilter(filter) {
    const filterType = detectFilterType(filter);
    if (filterType === FilterType.NETWORK) {
        return NetworkFilter.parse(filter, true);
    }
    else if (filterType === FilterType.COSMETIC) {
        return CosmeticFilter.parse(filter, true);
    }
    return null;
}
function parseFilters(list, config = new Config()) {
    config = new Config(config);
    const networkFilters = [];
    const cosmeticFilters = [];
    const notSupportedFilters = [];
    const lines = list.split('\n');
    const preprocessors = [];
    const preprocessorStack = [];
    for (let i = 0; i < lines.length; i += 1) {
        let line = lines[i];
        // Check if `line` should be left-trimmed
        if (line.length !== 0 && line.charCodeAt(0) <= 32) {
            line = line.trim();
        }
        // Handle continuations
        if (line.length > 2) {
            while (i < lines.length - 1 &&
                line.charCodeAt(line.length - 1) === 92 &&
                line.charCodeAt(line.length - 2) === 32) {
                line = line.slice(0, -2);
                const nextLine = lines[i + 1];
                if (nextLine.length > 4 &&
                    nextLine.charCodeAt(0) === 32 &&
                    nextLine.charCodeAt(1) === 32 &&
                    nextLine.charCodeAt(2) === 32 &&
                    nextLine.charCodeAt(3) === 32 &&
                    nextLine.charCodeAt(4) !== 32) {
                    line += nextLine.slice(4);
                    i += 1;
                }
                else {
                    break;
                }
            }
        }
        // Check if `line` should be right-trimmed
        if (line.length !== 0 && line.charCodeAt(line.length - 1) <= 32) {
            line = line.trim();
        }
        // Detect if filter is supported, network or cosmetic
        const filterType = detectFilterType(line, { extendedNonSupportedTypes: true });
        if (filterType === FilterType.NETWORK && config.loadNetworkFilters === true) {
            const filter = NetworkFilter.parse(line, config.debug);
            if (filter !== null) {
                networkFilters.push(filter);
                if (preprocessorStack.length > 0) {
                    preprocessorStack[preprocessorStack.length - 1].filterIDs.add(filter.getId());
                }
            }
            else {
                notSupportedFilters.push({
                    lineNumber: i,
                    filter: line,
                    filterType,
                });
            }
        }
        else if (filterType === FilterType.COSMETIC && config.loadCosmeticFilters === true) {
            const filter = CosmeticFilter.parse(line, config.debug);
            if (filter !== null) {
                if (config.loadGenericCosmeticsFilters === true || filter.isGenericHide() === false) {
                    cosmeticFilters.push(filter);
                    if (preprocessorStack.length > 0) {
                        preprocessorStack[preprocessorStack.length - 1].filterIDs.add(filter.getId());
                    }
                }
            }
            else {
                notSupportedFilters.push({
                    lineNumber: i,
                    filter: line,
                    filterType: FilterType.COSMETIC,
                });
            }
        }
        else if (config.loadPreprocessors) {
            const preprocessorToken = detectPreprocessor(line);
            if (preprocessorToken === PreprocessorTokens.BEGIF) {
                if (preprocessorStack.length > 0) {
                    preprocessorStack.push(new Preprocessor({
                        condition: `(${preprocessorStack[preprocessorStack.length - 1].condition})&&(${Preprocessor.getCondition(line)})`,
                    }));
                }
                else {
                    preprocessorStack.push(Preprocessor.parse(line));
                }
            }
            else if ((preprocessorToken === PreprocessorTokens.ENDIF ||
                preprocessorToken === PreprocessorTokens.ELSE) &&
                preprocessorStack.length > 0) {
                const lastPreprocessor = preprocessorStack.pop();
                preprocessors.push(lastPreprocessor);
                if (preprocessorToken === PreprocessorTokens.ELSE) {
                    preprocessorStack.push(new Preprocessor({
                        condition: `!(${lastPreprocessor.condition})`,
                    }));
                }
            }
            else if (filterType === FilterType.NOT_SUPPORTED_ADGUARD) {
                notSupportedFilters.push({
                    lineNumber: i,
                    filter: line,
                    filterType,
                });
            }
        }
        else if (filterType === FilterType.NOT_SUPPORTED_ADGUARD) {
            notSupportedFilters.push({
                lineNumber: i,
                filter: line,
                filterType,
            });
        }
    }
    return {
        networkFilters,
        cosmeticFilters,
        preprocessors: preprocessors.filter((preprocessor) => preprocessor.filterIDs.size > 0),
        notSupportedFilters,
    };
}
function getFilters(list, config) {
    const { networkFilters, cosmeticFilters, preprocessors } = parseFilters(list, config);
    const filters = [];
    return {
        filters: filters.concat(networkFilters).concat(cosmeticFilters),
        preprocessors,
    };
}
/**
 * Helper used to return a set of lines as strings where each line is
 * guaranteed to be a valid filter (i.e.: comments, empty lines and
 * un-supported filters are dropped).
 */
function getLinesWithFilters(list, config = new Config()) {
    // Set config to `debug` so that we keep track of raw lines for each filter
    return new Set(getFilters(list, new Config(Object.assign({}, config, { debug: true }))).filters.map(({ rawLine }) => rawLine));
}
/**
 * Merge several raw diffs into one, taking care of accumulating added and
 * removed filters, even if several diffs add/remove the same ones.
 */
function mergeDiffs(diffs) {
    const addedCumul = new Set();
    const removedCumul = new Set();
    const preprocessorsCumul = {};
    for (const { added, removed, preprocessors } of diffs) {
        if (added !== undefined) {
            for (const str of added) {
                if (removedCumul.has(str)) {
                    removedCumul.delete(str);
                }
                addedCumul.add(str);
            }
        }
        if (removed !== undefined) {
            for (const str of removed) {
                if (addedCumul.has(str)) {
                    addedCumul.delete(str);
                }
                removedCumul.add(str);
            }
        }
        if (!preprocessors) {
            continue;
        }
        for (const [condition, details] of Object.entries(preprocessors)) {
            if (!preprocessorsCumul[condition]) {
                preprocessorsCumul[condition] = {
                    added: details.added !== undefined ? new Set(details.added) : new Set(),
                    removed: details.removed !== undefined ? new Set(details.removed) : new Set(),
                };
            }
            else {
                if (details.added !== undefined) {
                    for (const str of details.added) {
                        if (preprocessorsCumul[condition].removed.has(str)) {
                            preprocessorsCumul[condition].removed.delete(str);
                        }
                        preprocessorsCumul[condition].added.add(str);
                    }
                }
                if (details.removed !== undefined) {
                    for (const str of details.removed) {
                        if (preprocessorsCumul[condition].added.has(str)) {
                            preprocessorsCumul[condition].added.delete(str);
                        }
                        preprocessorsCumul[condition].removed.add(str);
                    }
                }
            }
        }
    }
    return {
        added: Array.from(addedCumul),
        removed: Array.from(removedCumul),
        preprocessors: Object.fromEntries(Object.entries(preprocessorsCumul).map(([condition, details]) => [
            condition,
            {
                added: Array.from(details.added),
                removed: Array.from(details.removed),
            },
        ])),
    };
}

export { FilterType, detectFilterType, getLinesWithFilters, mergeDiffs, parseFilter, parseFilters };
