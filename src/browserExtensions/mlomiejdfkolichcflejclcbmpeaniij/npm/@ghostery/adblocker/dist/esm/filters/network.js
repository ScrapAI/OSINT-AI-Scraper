import { Domains } from '../engine/domains.js';
import { sizeOfUTF8, sizeOfNetworkFilter, sizeOfNetworkHostname, sizeOfRawNetwork, sizeOfNetworkCSP, sizeOfNetworkRedirect } from '../data-view.js';
import { toASCII } from '../punycode.js';
import { NORMALIZED_TYPE_TOKEN } from '../request.js';
import { TOKENS_BUFFER } from '../tokens-buffer.js';
import { fastHash, setBit, findLastIndexOfUnescapedCharacter, getBit, clearBit, fastStartsWithFrom, hasUnicode, bitCount, tokenizeWithWildcardsInPlace, tokenizeInPlace, tokenizeRegexInPlace, isDigit, isAlpha, HASH_INTERNAL_MULT, fastStartsWith, HASH_SEED } from '../utils.js';

/*!
 * Copyright (c) 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
const HTTP_HASH = fastHash('http');
const HTTPS_HASH = fastHash('https');
function isAllowedHostname(ch) {
    return (isDigit(ch) || isAlpha(ch) || ch === 95 /* '_' */ || ch === 45 /* '-' */ || ch === 46 /* '.' */);
}
/**
 * Masks used to store options of network filters in a bitmask.
 */
var NETWORK_FILTER_MASK;
(function (NETWORK_FILTER_MASK) {
    // Request Type
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromDocument"] = 1] = "fromDocument";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromFont"] = 2] = "fromFont";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromHttp"] = 4] = "fromHttp";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromHttps"] = 8] = "fromHttps";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromImage"] = 16] = "fromImage";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromMedia"] = 32] = "fromMedia";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromObject"] = 64] = "fromObject";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromOther"] = 128] = "fromOther";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromPing"] = 256] = "fromPing";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromScript"] = 512] = "fromScript";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromStylesheet"] = 1024] = "fromStylesheet";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromSubdocument"] = 2048] = "fromSubdocument";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromWebsocket"] = 4096] = "fromWebsocket";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["fromXmlHttpRequest"] = 8192] = "fromXmlHttpRequest";
    // Partiness
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["firstParty"] = 16384] = "firstParty";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["thirdParty"] = 32768] = "thirdParty";
    // Options
    // isReplace does not fit to the options, but is here from a lack of empty MASK slots
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isReplace"] = 65536] = "isReplace";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isBadFilter"] = 131072] = "isBadFilter";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isCSP"] = 262144] = "isCSP";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isGenericHide"] = 524288] = "isGenericHide";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isImportant"] = 1048576] = "isImportant";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isSpecificHide"] = 2097152] = "isSpecificHide";
    // Kind of patterns
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isFullRegex"] = 4194304] = "isFullRegex";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isRegex"] = 8388608] = "isRegex";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isUnicode"] = 16777216] = "isUnicode";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isLeftAnchor"] = 33554432] = "isLeftAnchor";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isRightAnchor"] = 67108864] = "isRightAnchor";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isException"] = 134217728] = "isException";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isHostnameAnchor"] = 268435456] = "isHostnameAnchor";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isRedirectRule"] = 536870912] = "isRedirectRule";
    NETWORK_FILTER_MASK[NETWORK_FILTER_MASK["isRedirect"] = 1073741824] = "isRedirect";
    // IMPORTANT: the mask is now full, no more options can be added
    // Consider creating a separate fitler type for isReplace if a new
    // network filter option is needed.
})(NETWORK_FILTER_MASK || (NETWORK_FILTER_MASK = {}));
/**
 * Mask used when a network filter can be applied on any content type.
 */
const FROM_ANY = NETWORK_FILTER_MASK.fromDocument |
    NETWORK_FILTER_MASK.fromFont |
    NETWORK_FILTER_MASK.fromImage |
    NETWORK_FILTER_MASK.fromMedia |
    NETWORK_FILTER_MASK.fromObject |
    NETWORK_FILTER_MASK.fromOther |
    NETWORK_FILTER_MASK.fromPing |
    NETWORK_FILTER_MASK.fromScript |
    NETWORK_FILTER_MASK.fromStylesheet |
    NETWORK_FILTER_MASK.fromSubdocument |
    NETWORK_FILTER_MASK.fromWebsocket |
    NETWORK_FILTER_MASK.fromXmlHttpRequest;
/**
 * Map content type value to mask the corresponding mask.
 * ref: https://developer.mozilla.org/en-US/docs/Mozilla/Tech/XPCOM/Reference/Interface/nsIContentPolicy
 */
const REQUEST_TYPE_TO_MASK = {
    beacon: NETWORK_FILTER_MASK.fromPing, // fromOther?
    document: NETWORK_FILTER_MASK.fromDocument,
    cspviolationreport: NETWORK_FILTER_MASK.fromOther,
    fetch: NETWORK_FILTER_MASK.fromXmlHttpRequest,
    font: NETWORK_FILTER_MASK.fromFont,
    image: NETWORK_FILTER_MASK.fromImage,
    imageset: NETWORK_FILTER_MASK.fromImage,
    mainFrame: NETWORK_FILTER_MASK.fromDocument,
    main_frame: NETWORK_FILTER_MASK.fromDocument,
    media: NETWORK_FILTER_MASK.fromMedia,
    object: NETWORK_FILTER_MASK.fromObject,
    object_subrequest: NETWORK_FILTER_MASK.fromObject,
    ping: NETWORK_FILTER_MASK.fromPing, // fromOther?
    script: NETWORK_FILTER_MASK.fromScript,
    stylesheet: NETWORK_FILTER_MASK.fromStylesheet,
    subFrame: NETWORK_FILTER_MASK.fromSubdocument,
    sub_frame: NETWORK_FILTER_MASK.fromSubdocument,
    webSocket: NETWORK_FILTER_MASK.fromWebsocket,
    websocket: NETWORK_FILTER_MASK.fromWebsocket,
    xhr: NETWORK_FILTER_MASK.fromXmlHttpRequest,
    xmlhttprequest: NETWORK_FILTER_MASK.fromXmlHttpRequest,
    // Other
    cspReport: NETWORK_FILTER_MASK.fromOther,
    csp_report: NETWORK_FILTER_MASK.fromOther,
    eventsource: NETWORK_FILTER_MASK.fromOther,
    manifest: NETWORK_FILTER_MASK.fromOther,
    other: NETWORK_FILTER_MASK.fromOther,
    prefetch: NETWORK_FILTER_MASK.fromOther,
    preflight: NETWORK_FILTER_MASK.fromOther,
    signedexchange: NETWORK_FILTER_MASK.fromOther,
    speculative: NETWORK_FILTER_MASK.fromOther,
    texttrack: NETWORK_FILTER_MASK.fromOther,
    web_manifest: NETWORK_FILTER_MASK.fromOther,
    xml_dtd: NETWORK_FILTER_MASK.fromOther,
    xslt: NETWORK_FILTER_MASK.fromOther,
};
function getListOfRequestTypesNegated(filter) {
    const types = [];
    if (filter.fromDocument() === false) {
        types.push('document');
    }
    if (filter.fromImage() === false) {
        types.push('image');
    }
    if (filter.fromMedia() === false) {
        types.push('media');
    }
    if (filter.fromObject() === false) {
        types.push('object');
    }
    if (filter.fromOther() === false) {
        types.push('other');
    }
    if (filter.fromPing() === false) {
        types.push('ping');
    }
    if (filter.fromScript() === false) {
        types.push('script');
    }
    if (filter.fromStylesheet() === false) {
        types.push('stylesheet');
    }
    if (filter.fromSubdocument() === false) {
        types.push('sub_frame');
    }
    if (filter.fromWebsocket() === false) {
        types.push('websocket');
    }
    if (filter.fromXmlHttpRequest() === false) {
        types.push('xhr');
    }
    if (filter.fromFont() === false) {
        types.push('font');
    }
    return types;
}
function getListOfRequestTypes(filter) {
    const types = [];
    if (filter.fromDocument()) {
        types.push('document');
    }
    if (filter.fromImage()) {
        types.push('image');
    }
    if (filter.fromMedia()) {
        types.push('media');
    }
    if (filter.fromObject()) {
        types.push('object');
    }
    if (filter.fromOther()) {
        types.push('other');
    }
    if (filter.fromPing()) {
        types.push('ping');
    }
    if (filter.fromScript()) {
        types.push('script');
    }
    if (filter.fromStylesheet()) {
        types.push('stylesheet');
    }
    if (filter.fromSubdocument()) {
        types.push('sub_frame');
    }
    if (filter.fromWebsocket()) {
        types.push('websocket');
    }
    if (filter.fromXmlHttpRequest()) {
        types.push('xhr');
    }
    if (filter.fromFont()) {
        types.push('font');
    }
    return types;
}
function computeFilterId(mask, filter, hostname, domains, denyallow, optionValue) {
    let hash = (HASH_SEED * HASH_INTERNAL_MULT) ^ mask;
    if (domains !== undefined) {
        hash = domains.updateId(hash);
    }
    if (denyallow !== undefined) {
        hash = denyallow.updateId(hash);
    }
    if (filter !== undefined) {
        for (let i = 0; i < filter.length; i += 1) {
            hash = (hash * HASH_INTERNAL_MULT) ^ filter.charCodeAt(i);
        }
    }
    if (hostname !== undefined) {
        for (let i = 0; i < hostname.length; i += 1) {
            hash = (hash * HASH_INTERNAL_MULT) ^ hostname.charCodeAt(i);
        }
    }
    if (optionValue !== undefined) {
        for (let i = 0; i < optionValue.length; i += 1) {
            hash = (hash * HASH_INTERNAL_MULT) ^ optionValue.charCodeAt(i);
        }
    }
    return hash >>> 0;
}
/**
 * Compiles a filter pattern to a regex. This is only performed *lazily* for
 * filters containing at least a * or ^ symbol. Because Regexes are expansive,
 * we try to convert some patterns to plain filters.
 */
function compileRegex(filter, isLeftAnchor, isRightAnchor, isFullRegex) {
    if (isFullRegex === true) {
        return new RegExp(filter.slice(1, filter.length - 1), 'i');
    }
    // Escape special regex characters: |.$+?{}()[]\
    filter = filter.replace(/([|.$+?{}()[\]\\])/g, '\\$1');
    // * can match anything
    filter = filter.replace(/\*/g, '.*');
    // ^ can match any separator or the end of the pattern
    filter = filter.replace(/\^/g, '(?:[^\\w\\d_.%-]|$)');
    // Should match end of url
    if (isRightAnchor) {
        filter = `${filter}$`;
    }
    if (isLeftAnchor) {
        filter = `^${filter}`;
    }
    return new RegExp(filter);
}
/**
 * Collects a filter option key until the function sees the special character.
 * This function will stop iterating over the given string if it sees equal sign or comma sign.
 * If there's an equal sign, it means that we'll see the value.
 * Otherwise, if there's a comma sign, it means that the option doesn't have any values.
 */
function getFilterOptionName(line, pos, end) {
    const start = pos;
    for (; pos < end; pos++) {
        const code = line.charCodeAt(pos);
        if (code === 61 /* '=' */ || code === 44 /* ',' */) {
            end = pos;
            break;
        }
    }
    return [pos, line.slice(start, end)];
}
/**
 * Collects a filter option value until the function sees the special character.
 * This function respects the escaping characters, so we can safely collect the full value
 * including the special characters which are not allowed normally.
 * This function will stop if it sees a comma sign.
 */
function getFilterOptionValue(line, pos, end) {
    let start = pos;
    let value = '';
    for (; pos < end; pos++) {
        const code = line.charCodeAt(pos);
        if (code === 92 /* '\\' */) {
            value += line.slice(start, pos);
            start = ++pos;
        }
        else if (code === 44 /* ',' */) {
            break;
        }
    }
    if (start - pos !== 0) {
        value += line.slice(start, pos);
    }
    return [pos, value];
}
const REGEXP_CHARACTER_ESCAPES = new Set([
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Character_escape
    102, // f
    110, // n
    114, // r
    116, // t
    118, // v
    48, // 0
    94, // ^
    36, // $
    92, // \
    46, // .
    42, // *
    43, // +
    63, // ?
    40, // (
    41, // )
    91, // [
    93, // ]
    123, // {
    125, // }
    124, // |
    47, // /
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Character_class_escape
    100, // d
    68, // D
    119, // s
    87, // S
    115, // w
    83, // W
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_expressions/Word_boundary_assertion
    98, // b
    66, // B
]);
function isHexLiteral(code) {
    return ((code >= 48 && code <= 57) || // 0-9
        (code <= 65 && code <= 70) || // A-F
        (code >= 97 && code <= 102) // a-f
    );
}
/**
 * Checks if the character(s) after the escape character should be preserved.
 * @returns The end index of search range and boolean indicating whether the character escape was found or not.
 */
function isCharacterEscapeInRegExp(line, pos, _end) {
    const code = line.charCodeAt(pos + 1);
    if (code === 44 /* ',' */ || REGEXP_CHARACTER_ESCAPES.has(code)) {
        return [pos + 1, true];
    }
    // \cA, \cB, …, \cz
    if (code === 99 /* 'c' */) {
        const next = line.charCodeAt(pos + 2);
        if ((next >= 65 && next <= 90) || (next >= 97 && next <= 122)) {
            return [pos + 2, true];
        }
    }
    // \xHH
    if (code === 120 /* 'x' */ &&
        isHexLiteral(line.charCodeAt(pos + 2)) &&
        isHexLiteral(line.charCodeAt(pos + 3))) {
        return [pos + 3, true];
    }
    if (code === 117 /* 'u' */) {
        // \u{HHH}
        if (line.charCodeAt(pos + 2) === 123 /* '{' */) {
            const close = line.indexOf('}', pos + 3);
            const hexLiteralLength = close - pos + 3;
            if (hexLiteralLength >= 1 && hexLiteralLength <= 6) {
                return [close, true];
            }
        }
        else if (
        // \uHHHH
        isHexLiteral(line.charCodeAt(pos + 2)) &&
            isHexLiteral(line.charCodeAt(pos + 3)) &&
            isHexLiteral(line.charCodeAt(pos + 4)) &&
            isHexLiteral(line.charCodeAt(pos + 5))) {
            return [pos + 5, true];
        }
    }
    return [pos + 1, false];
}
/**
 * Comma and slash are expected to come with escape character by the spec.
 * https://adguard.com/kb/general/ad-filtering/create-own-filters/#replace-modifier
 */
function isCharacterEscapeInReplace(line, pos, end) {
    const code = line.charCodeAt(pos + 1);
    if (code === 44 /* ',' */ || code === 47 /* '/' */) {
        return [pos + 1, false];
    }
    return isCharacterEscapeInRegExp(line, pos);
}
/**
 * Collects a filter option value of the replace modifier.
 * This function respects the escaping character with the allowed characters of the replace modifier.
 * In the replace modifier, it can include the any sign allowed in the regular expression.
 * Therefore, a comma sign can interfere the `getFilterOptionValue` function.
 * This function will not stop unless it collects all the parts of the replace modifier option value.
 */
function getFilterReplaceOptionValue(line, pos, end) {
    // Try to fast exit if the first character is an unexpected character.
    if (line.charCodeAt(pos++) !== 47 /* '/' */) {
        return [end, undefined];
    }
    const parts = ['', '', ''];
    let start = pos;
    let slashes = 0;
    for (; pos < end; pos++) {
        const code = line.charCodeAt(pos);
        if (code === 92 /* '\\' */) {
            parts[slashes] += line.slice(start, pos);
            const [posAfterCharacterEscape, isCharacterEscape] = isCharacterEscapeInReplace(line, pos);
            if (isCharacterEscape === false) {
                // Remove escaping character ('\\') by adding an offset to next `start` assignment.
                ++pos;
            }
            start = pos;
            pos = posAfterCharacterEscape;
        }
        else if (code === 47 /* '/' */) {
            if (pos - start !== 0) {
                parts[slashes] += line.slice(start, pos);
            }
            start = pos + 1;
            if (++slashes === 2) {
                // Since we saw 3 slashes in total, it means that the option value should be closed here.
                // Note that we already saw the first slash before the loop.
                break;
            }
        }
    }
    const valueEnd = line.indexOf(',', pos);
    if (valueEnd !== -1) {
        end = valueEnd;
    }
    parts[2] = line.slice(start, end);
    pos = end;
    return [pos, parts];
}
/**
 * Collects an array of filter options from the given index.
 * This function leverages `getFilterOptionKey`, `getFilterOptionValue`, and every extension functions.
 * Depending on the filter option key, the function to collect filter option value can vary.
 * For the generic filter option value, it'll use `getFilterOptionValue` function to get the value.
 */
function getFilterOptions(line, pos, end) {
    const options = [];
    let name;
    let value;
    for (; pos < end; pos++) {
        [pos, name] = getFilterOptionName(line, pos, end);
        if (name !== undefined) {
            if (line.charCodeAt(pos) === 61 /* '=' */) {
                pos++;
            }
            if (name === 'replace') {
                const result = getFilterReplaceOptionValue(line, pos, end);
                if (result[1] === undefined) {
                    value = '';
                }
                else {
                    value = line.slice(pos, result[0]);
                }
                pos = result[0];
            }
            else {
                [pos, value] = getFilterOptionValue(line, pos, end);
            }
            options.push([name, value]);
        }
    }
    return options;
}
/**
 * Transforms the replace modifier option value into the regular expression with its replacement.
 * This function takes a fixed length array from `getFilterReplaceOptionValue`
 * then try to build the regular expression.
 * This function will return `null` if the array format or the given regular expression components are not valid.
 */
function replaceOptionValueToRegexp(value) {
    const [, values] = getFilterReplaceOptionValue(value, 0, value.length);
    if (values === undefined) {
        return null;
    }
    // RegExp constructor can throw an error
    try {
        // We expect `/regexp/replacement/flags` to be [regexp, replacement, flags]
        // The first slash should be removed in the early steps
        return [new RegExp(values[0], values[2]), values[1]];
    }
    catch (error) {
        return null;
    }
}
const MATCH_ALL = new RegExp('');
class NetworkFilter {
    static parse(line, debug = false) {
        // Represent options as a bitmask
        let mask = NETWORK_FILTER_MASK.thirdParty |
            NETWORK_FILTER_MASK.firstParty |
            NETWORK_FILTER_MASK.fromHttps |
            NETWORK_FILTER_MASK.fromHttp;
        // Temporary masks for positive (e.g.: $script) and negative (e.g.: $~script)
        // content type options.
        let cptMaskPositive = 0;
        let cptMaskNegative = FROM_ANY;
        let hostname;
        let domains;
        let denyallow;
        let optionValue;
        // Start parsing
        let filterIndexStart = 0;
        let filterIndexEnd = line.length;
        // @@filter == Exception
        if (line.charCodeAt(0) === 64 /* '@' */ && line.charCodeAt(1) === 64 /* '@' */) {
            filterIndexStart += 2;
            mask = setBit(mask, NETWORK_FILTER_MASK.isException);
        }
        // filter$options == Options
        // ^     ^
        // |     |
        // |     optionsIndex
        // filterIndexStart
        const optionsIndex = findLastIndexOfUnescapedCharacter(line, '$');
        if (optionsIndex !== -1 && line.charCodeAt(optionsIndex + 1) !== 47 /* '/' */) {
            // Parse options and set flags
            filterIndexEnd = optionsIndex;
            // --------------------------------------------------------------------- //
            // parseOptions
            // --------------------------------------------------------------------- //
            for (const rawOption of getFilterOptions(line, optionsIndex + 1, line.length)) {
                const negation = rawOption[0].charCodeAt(0) === 126; /* '~' */
                const option = negation === true ? rawOption[0].slice(1) : rawOption[0];
                const value = rawOption[1];
                switch (option) {
                    case 'denyallow': {
                        denyallow = Domains.parse(value.split('|'), debug);
                        break;
                    }
                    case 'domain':
                    case 'from': {
                        // domain list starting or ending with '|' is invalid
                        if (value.charCodeAt(0) === 124 /* '|' */ ||
                            value.charCodeAt(value.length - 1) === 124 /* '|' */) {
                            return null;
                        }
                        domains = Domains.parse(value.split('|'), debug);
                        break;
                    }
                    case 'badfilter':
                        mask = setBit(mask, NETWORK_FILTER_MASK.isBadFilter);
                        break;
                    case 'important':
                        // Note: `negation` should always be `false` here.
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isImportant);
                        break;
                    case 'match-case':
                        // Note: `negation` should always be `false` here.
                        if (negation) {
                            return null;
                        }
                        // We currently consider all filters to be case-insensitive.
                        break;
                    case '3p':
                    case 'third-party':
                        if (negation) {
                            // ~third-party means we should clear the flag
                            mask = clearBit(mask, NETWORK_FILTER_MASK.thirdParty);
                        }
                        else {
                            // third-party means ~first-party
                            mask = clearBit(mask, NETWORK_FILTER_MASK.firstParty);
                        }
                        break;
                    case '1p':
                    case 'first-party':
                        if (negation) {
                            // ~first-party means we should clear the flag
                            mask = clearBit(mask, NETWORK_FILTER_MASK.firstParty);
                        }
                        else {
                            // first-party means ~third-party
                            mask = clearBit(mask, NETWORK_FILTER_MASK.thirdParty);
                        }
                        break;
                    case 'redirect-rule':
                    case 'redirect': {
                        // Negation of redirection doesn't make sense
                        if (negation) {
                            return null;
                        }
                        // Ignore this filter if no redirection resource is specified
                        if (value.length === 0) {
                            return null;
                        }
                        // Ignore this filter if wrong priority is given
                        const priorityIndex = value.lastIndexOf(':');
                        if (priorityIndex === 0) {
                            return null;
                        }
                        else if (priorityIndex !== -1 &&
                            (isNaN(Number(value.slice(priorityIndex + 1))) === true ||
                                priorityIndex + 1 === value.length)) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isRedirect);
                        if (option === 'redirect-rule') {
                            mask = setBit(mask, NETWORK_FILTER_MASK.isRedirectRule);
                        }
                        optionValue = value;
                        break;
                    }
                    case 'csp':
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isCSP);
                        if (value.length > 0) {
                            optionValue = value;
                        }
                        break;
                    case 'ehide':
                    case 'elemhide':
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isGenericHide);
                        mask = setBit(mask, NETWORK_FILTER_MASK.isSpecificHide);
                        break;
                    case 'shide':
                    case 'specifichide':
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isSpecificHide);
                        break;
                    case 'ghide':
                    case 'generichide':
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isGenericHide);
                        break;
                    case 'inline-script':
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isCSP);
                        optionValue =
                            "script-src 'self' 'unsafe-eval' http: https: data: blob: mediastream: filesystem:";
                        break;
                    case 'inline-font':
                        if (negation) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isCSP);
                        optionValue =
                            "font-src 'self' 'unsafe-eval' http: https: data: blob: mediastream: filesystem:";
                        break;
                    case 'replace':
                    case 'content':
                        if (negation ||
                            (value.length === 0
                                ? getBit(mask, NETWORK_FILTER_MASK.isException) === false
                                : replaceOptionValueToRegexp(value) === null)) {
                            return null;
                        }
                        mask = setBit(mask, NETWORK_FILTER_MASK.isReplace);
                        optionValue = value;
                        break;
                    default: {
                        // Handle content type options separatly
                        let optionMask = 0;
                        switch (option) {
                            case 'all':
                                if (negation) {
                                    return null;
                                }
                                // NOTE: Currently a filter cannot be both blocking and CSP, so
                                // we will have to create multiple filters to keep the semantics
                                // of 'all'.
                                // mask = setBit(mask, NETWORK_FILTER_MASK.isCSP);
                                // csp = [
                                //   "script-src 'self' 'unsafe-eval' http: https: data: blob: mediastream: filesystem:",
                                //   "font-src 'self' 'unsafe-eval' http: https: data: blob: mediastream: filesystem:",
                                // ].join('; ');
                                break;
                            case 'image':
                                optionMask = NETWORK_FILTER_MASK.fromImage;
                                break;
                            case 'media':
                                optionMask = NETWORK_FILTER_MASK.fromMedia;
                                break;
                            case 'object':
                            case 'object-subrequest':
                                optionMask = NETWORK_FILTER_MASK.fromObject;
                                break;
                            case 'other':
                                optionMask = NETWORK_FILTER_MASK.fromOther;
                                break;
                            case 'ping':
                            case 'beacon':
                                optionMask = NETWORK_FILTER_MASK.fromPing;
                                break;
                            case 'script':
                                optionMask = NETWORK_FILTER_MASK.fromScript;
                                break;
                            case 'css':
                            case 'stylesheet':
                                optionMask = NETWORK_FILTER_MASK.fromStylesheet;
                                break;
                            case 'frame':
                            case 'subdocument':
                                optionMask = NETWORK_FILTER_MASK.fromSubdocument;
                                break;
                            case 'xhr':
                            case 'xmlhttprequest':
                                optionMask = NETWORK_FILTER_MASK.fromXmlHttpRequest;
                                break;
                            case 'websocket':
                                optionMask = NETWORK_FILTER_MASK.fromWebsocket;
                                break;
                            case 'font':
                                optionMask = NETWORK_FILTER_MASK.fromFont;
                                break;
                            case 'doc':
                            case 'document':
                                optionMask = NETWORK_FILTER_MASK.fromDocument;
                                break;
                            default:
                                // Disable this filter if we don't support all the options
                                return null;
                        }
                        // We got a valid cpt option, update mask
                        if (negation) {
                            cptMaskNegative = clearBit(cptMaskNegative, optionMask);
                        }
                        else {
                            cptMaskPositive = setBit(cptMaskPositive, optionMask);
                        }
                        break;
                    }
                }
            }
            // End of option parsing
            // --------------------------------------------------------------------- //
        }
        if (cptMaskPositive === 0) {
            mask |= cptMaskNegative;
        }
        else if (cptMaskNegative === FROM_ANY) {
            mask |= cptMaskPositive;
        }
        else {
            mask |= cptMaskPositive & cptMaskNegative;
        }
        // Identify kind of pattern
        let filter;
        // Detect Regexps (i.e.: /pattern/)
        if (filterIndexEnd - filterIndexStart >= 2 &&
            line.charCodeAt(filterIndexStart) === 47 /* '/' */ &&
            line.charCodeAt(filterIndexEnd - 1) === 47 /* '/' */) {
            // Some extra ideas which could be applied to RegExp filters:
            // * convert rules without any special RegExp syntax to plain patterns
            // * remove extra `isFullRegex` flag since `isRegex` might be enough
            // * apply some optimizations on the fly: /^https?:\\/\\/rest => isHttp + isHttps + rest
            filter = line.slice(filterIndexStart, filterIndexEnd);
            // Validate RegExp to make sure this rule is fine
            try {
                compileRegex(filter, false /* isLeftAnchor */, false /* isRightAnchor */, true /* isFullRegex */);
            }
            catch (ex) {
                return null; // invalid RegExp
            }
            mask = setBit(mask, NETWORK_FILTER_MASK.isFullRegex);
        }
        else {
            // Deal with hostname pattern
            if (filterIndexEnd > 0 && line.charCodeAt(filterIndexEnd - 1) === 124 /* '|' */) {
                mask = setBit(mask, NETWORK_FILTER_MASK.isRightAnchor);
                filterIndexEnd -= 1;
            }
            if (filterIndexStart < filterIndexEnd &&
                line.charCodeAt(filterIndexStart) === 124 /* '|' */) {
                if (filterIndexStart < filterIndexEnd - 1 &&
                    line.charCodeAt(filterIndexStart + 1) === 124 /* '|' */) {
                    mask = setBit(mask, NETWORK_FILTER_MASK.isHostnameAnchor);
                    filterIndexStart += 2;
                }
                else {
                    mask = setBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    filterIndexStart += 1;
                }
            }
            // const isRegex = checkIsRegex(line, filterIndexStart, filterIndexEnd);
            // mask = setNetworkMask(mask, NETWORK_FILTER_MASK.isRegex, isRegex);
            if (getBit(mask, NETWORK_FILTER_MASK.isHostnameAnchor)) {
                // Split at the first character which is not allowed in a hostname
                let firstSeparator = filterIndexStart;
                while (firstSeparator < filterIndexEnd &&
                    isAllowedHostname(line.charCodeAt(firstSeparator)) === true) {
                    firstSeparator += 1;
                }
                // No separator found so hostname has full length
                if (firstSeparator === filterIndexEnd) {
                    hostname = line.slice(filterIndexStart, filterIndexEnd);
                    filterIndexStart = filterIndexEnd;
                    // mask = setBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                }
                else {
                    // Found a separator
                    hostname = line.slice(filterIndexStart, firstSeparator);
                    filterIndexStart = firstSeparator;
                    const separatorCode = line.charCodeAt(firstSeparator);
                    if (separatorCode === 94 /* '^' */) {
                        // If the only symbol remaining for the selector is '^' then ignore it
                        // but set the filter as right anchored since there should not be any
                        // other label on the right
                        if (filterIndexEnd - filterIndexStart === 1) {
                            filterIndexStart = filterIndexEnd;
                            mask = setBit(mask, NETWORK_FILTER_MASK.isRightAnchor);
                        }
                        else {
                            mask = setBit(mask, NETWORK_FILTER_MASK.isRegex);
                            mask = setBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                        }
                    }
                    else if (separatorCode === 42 /* '*' */) {
                        mask = setBit(mask, NETWORK_FILTER_MASK.isRegex);
                        // mask = setBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    }
                    else {
                        mask = setBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    }
                }
            }
            // Remove trailing '*'
            if (filterIndexEnd - filterIndexStart > 0 &&
                line.charCodeAt(filterIndexEnd - 1) === 42 /* '*' */) {
                filterIndexEnd -= 1;
            }
            // Remove leading '*' if the filter is not hostname anchored.
            if (getBit(mask, NETWORK_FILTER_MASK.isHostnameAnchor) === false &&
                filterIndexEnd - filterIndexStart > 0 &&
                line.charCodeAt(filterIndexStart) === 42 /* '*' */) {
                mask = clearBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                filterIndexStart += 1;
            }
            // Transform filters on protocol (http, https, ws)
            if (getBit(mask, NETWORK_FILTER_MASK.isLeftAnchor)) {
                if (filterIndexEnd - filterIndexStart === 5 &&
                    fastStartsWithFrom(line, 'ws://', filterIndexStart)) {
                    mask = setBit(mask, NETWORK_FILTER_MASK.fromWebsocket);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.fromHttp);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.fromHttps);
                    filterIndexStart = filterIndexEnd;
                }
                else if (filterIndexEnd - filterIndexStart === 7 &&
                    fastStartsWithFrom(line, 'http://', filterIndexStart)) {
                    mask = setBit(mask, NETWORK_FILTER_MASK.fromHttp);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.fromHttps);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    filterIndexStart = filterIndexEnd;
                }
                else if (filterIndexEnd - filterIndexStart === 8 &&
                    fastStartsWithFrom(line, 'https://', filterIndexStart)) {
                    mask = setBit(mask, NETWORK_FILTER_MASK.fromHttps);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.fromHttp);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    filterIndexStart = filterIndexEnd;
                }
                else if (filterIndexEnd - filterIndexStart === 8 &&
                    fastStartsWithFrom(line, 'http*://', filterIndexStart)) {
                    mask = setBit(mask, NETWORK_FILTER_MASK.fromHttps);
                    mask = setBit(mask, NETWORK_FILTER_MASK.fromHttp);
                    mask = clearBit(mask, NETWORK_FILTER_MASK.isLeftAnchor);
                    filterIndexStart = filterIndexEnd;
                }
            }
            if (filterIndexEnd - filterIndexStart > 0) {
                filter = line.slice(filterIndexStart, filterIndexEnd).toLowerCase();
                mask = setNetworkMask(mask, NETWORK_FILTER_MASK.isUnicode, hasUnicode(filter));
                if (getBit(mask, NETWORK_FILTER_MASK.isRegex) === false) {
                    mask = setNetworkMask(mask, NETWORK_FILTER_MASK.isRegex, checkIsRegex(filter, 0, filter.length));
                }
            }
            // TODO
            // - ignore hostname anchor is not hostname provided
            if (hostname !== undefined) {
                hostname = hostname.toLowerCase();
                if (hasUnicode(hostname)) {
                    mask = setNetworkMask(mask, NETWORK_FILTER_MASK.isUnicode, true);
                    hostname = toASCII(hostname);
                }
            }
        }
        return new NetworkFilter({
            filter,
            hostname,
            mask,
            domains,
            denyallow,
            optionValue,
            rawLine: debug === true ? line : undefined,
            regex: undefined,
        });
    }
    /**
     * Deserialize network filters. The code accessing the buffer should be
     * symetrical to the one in `serializeNetworkFilter`.
     */
    static deserialize(buffer) {
        const mask = buffer.getUint32();
        const optionalParts = buffer.getUint8();
        const isUnicode = getBit(mask, NETWORK_FILTER_MASK.isUnicode);
        // The order of these statements is important. Since `buffer.getX()` will
        // internally increment the position of next byte to read, they need to be
        // retrieved in the exact same order they were serialized (check
        // `serializeNetworkFilter`).
        return new NetworkFilter({
            // Mandatory field
            mask,
            // Optional parts
            filter: (optionalParts & 1) === 1
                ? isUnicode
                    ? buffer.getUTF8()
                    : buffer.getNetworkFilter()
                : undefined,
            hostname: (optionalParts & 2) === 2 ? buffer.getNetworkHostname() : undefined,
            domains: (optionalParts & 4) === 4 ? Domains.deserialize(buffer) : undefined,
            rawLine: (optionalParts & 8) === 8 ? buffer.getRawNetwork() : undefined,
            denyallow: (optionalParts & 16) === 16 ? Domains.deserialize(buffer) : undefined,
            optionValue: (optionalParts & 32) === 32
                ? getBit(mask, NETWORK_FILTER_MASK.isCSP)
                    ? buffer.getNetworkCSP()
                    : getBit(mask, NETWORK_FILTER_MASK.isRedirect)
                        ? buffer.getNetworkRedirect()
                        : buffer.getUTF8()
                : undefined,
            regex: undefined,
        });
    }
    constructor({ filter, hostname, mask, domains, denyallow, optionValue, rawLine, regex, }) {
        this.filter = filter;
        this.hostname = hostname;
        this.mask = mask;
        this.domains = domains;
        this.denyallow = denyallow;
        this.optionValue = optionValue;
        this.rawLine = rawLine;
        this.id = undefined;
        this.regex = regex;
    }
    get csp() {
        if (!this.isCSP()) {
            return undefined;
        }
        return this.optionValue;
    }
    get redirect() {
        if (!this.isRedirect()) {
            return undefined;
        }
        return this.optionValue;
    }
    isCosmeticFilter() {
        return false;
    }
    isNetworkFilter() {
        return true;
    }
    match(request) {
        return checkOptions(this, request) && checkPattern(this, request);
    }
    /**
     * To allow for a more compact representation of network filters, the
     * representation is composed of a mandatory header, and some optional
     *
     * Header:
     * =======
     *
     *  | opt | mask
     *     8     32
     *
     * For an empty filter having no pattern, hostname, the minimum size is: 42 bits.
     *
     * Then for each optional part (filter, hostname optDomains, optNotDomains,
     * redirect), it takes 16 bits for the length of the string + the length of the
     * string in bytes.
     *
     * The optional parts are written in order of there number of occurrence in the
     * filter list used by the adblocker. The most common being `hostname`, then
     * `filter`, `optDomains`, `optNotDomains`, `redirect`.
     *
     * Example:
     * ========
     *
     * @@||cliqz.com would result in a serialized version:
     *
     * | 1 | mask | 9 | c | l | i | q | z | . | c | o | m  (16 bytes)
     *
     * In this case, the serialized version is actually bigger than the original
     * filter, but faster to deserialize. In the future, we could optimize the
     * representation to compact small filters better.
     *
     * Ideas:
     *  * variable length encoding for the mask (if not option, take max 1 byte).
     *  * first byte could contain the mask as well if small enough.
     *  * when packing ascii string, store several of them in each byte.
     */
    serialize(buffer) {
        buffer.pushUint32(this.mask);
        const index = buffer.getPos();
        buffer.pushUint8(0);
        // This bit-mask indicates which optional parts of the filter were serialized.
        let optionalParts = 0;
        if (this.filter !== undefined) {
            optionalParts |= 1;
            if (this.isUnicode()) {
                buffer.pushUTF8(this.filter);
            }
            else {
                buffer.pushNetworkFilter(this.filter);
            }
        }
        if (this.hostname !== undefined) {
            optionalParts |= 2;
            buffer.pushNetworkHostname(this.hostname);
        }
        if (this.domains !== undefined) {
            optionalParts |= 4;
            this.domains.serialize(buffer);
        }
        if (this.rawLine !== undefined) {
            optionalParts |= 8;
            buffer.pushRawNetwork(this.rawLine);
        }
        if (this.denyallow !== undefined) {
            optionalParts |= 16;
            this.denyallow.serialize(buffer);
        }
        if (this.optionValue !== undefined) {
            optionalParts |= 32;
            if (this.isCSP()) {
                buffer.pushNetworkCSP(this.optionValue);
            }
            else if (this.isRedirect()) {
                buffer.pushNetworkRedirect(this.optionValue);
            }
            else {
                buffer.pushUTF8(this.optionValue);
            }
        }
        buffer.setByte(index, optionalParts);
    }
    getSerializedSize(compression) {
        let estimate = 4 + 1; // mask = 4 bytes // optional parts = 1 byte
        if (this.filter !== undefined) {
            if (this.isUnicode() === true) {
                estimate += sizeOfUTF8(this.filter);
            }
            else {
                estimate += sizeOfNetworkFilter(this.filter, compression);
            }
        }
        if (this.hostname !== undefined) {
            estimate += sizeOfNetworkHostname(this.hostname, compression);
        }
        if (this.domains !== undefined) {
            estimate += this.domains.getSerializedSize();
        }
        if (this.rawLine !== undefined) {
            estimate += sizeOfRawNetwork(this.rawLine, compression);
        }
        if (this.denyallow !== undefined) {
            estimate += this.denyallow.getSerializedSize();
        }
        if (this.optionValue !== undefined) {
            if (this.isCSP()) {
                estimate += sizeOfNetworkCSP(this.optionValue, compression);
            }
            else if (this.isRedirect()) {
                estimate += sizeOfNetworkRedirect(this.optionValue, compression);
            }
            else {
                estimate += sizeOfUTF8(this.optionValue);
            }
        }
        return estimate;
    }
    /**
     * Tries to recreate the original representation of the filter (adblock
     * syntax) from the internal representation. If `rawLine` is set (when filters
     * are parsed in `debug` mode for example), then it is returned directly.
     * Otherwise, we try to stick as closely as possible to the original form;
     * there are things which cannot be recovered though, like domains options
     * of which only hashes are stored.
     */
    toString(modifierReplacer) {
        if (this.rawLine !== undefined) {
            return this.rawLine;
        }
        let filter = '';
        if (this.isException()) {
            filter += '@@';
        }
        if (this.isHostnameAnchor()) {
            filter += '||';
        }
        else if (this.fromHttp() !== this.fromHttps()) {
            if (this.fromHttp()) {
                filter += '|http://';
            }
            else {
                filter += '|https://';
            }
        }
        else if (this.isLeftAnchor()) {
            filter += '|';
        }
        if (this.hasHostname()) {
            filter += this.getHostname();
            filter += '^';
        }
        if (this.isFullRegex()) {
            filter += `/${this.getRegex().source}/`;
        }
        else if (this.isRegex()) {
            filter += this.getRegex().source;
        }
        else {
            filter += this.getFilter();
        }
        if (this.isRightAnchor() && filter[filter.length - 1] !== '^') {
            filter += '|';
        }
        // Options
        const options = [];
        if (this.fromAny() === false) {
            const numberOfCptOptions = bitCount(this.getCptMask());
            const numberOfNegatedOptions = bitCount(FROM_ANY) - numberOfCptOptions;
            if (numberOfNegatedOptions < numberOfCptOptions) {
                for (const type of getListOfRequestTypesNegated(this)) {
                    options.push(`~${type}`);
                }
            }
            else {
                for (const type of getListOfRequestTypes(this)) {
                    options.push(type);
                }
            }
        }
        if (this.isImportant()) {
            options.push('important');
        }
        if (this.isRedirectRule()) {
            if (this.optionValue === '') {
                options.push('redirect-rule');
            }
            else {
                options.push(`redirect-rule=${this.optionValue}`);
            }
        }
        else if (this.isRedirect()) {
            if (this.optionValue === '') {
                options.push('redirect');
            }
            else {
                options.push(`redirect=${this.optionValue}`);
            }
        }
        if (this.isCSP()) {
            options.push(`csp=${this.optionValue}`);
        }
        if (this.isElemHide()) {
            options.push('elemhide');
        }
        if (this.isSpecificHide()) {
            options.push('specifichide');
        }
        if (this.isGenericHide()) {
            options.push('generichide');
        }
        if (this.firstParty() !== this.thirdParty()) {
            if (this.firstParty()) {
                options.push('1p');
            }
            if (this.thirdParty()) {
                options.push('3p');
            }
        }
        if (this.domains !== undefined) {
            if (this.domains.parts !== undefined) {
                options.push(`domain=${this.domains.parts}`);
            }
            else {
                options.push('domain=<hashed>');
            }
        }
        if (this.denyallow !== undefined) {
            if (this.denyallow.parts !== undefined) {
                options.push(`denyallow=${this.denyallow.parts}`);
            }
            else {
                options.push('denyallow=<hashed>');
            }
        }
        if (this.isBadFilter()) {
            options.push('badfilter');
        }
        if (options.length > 0) {
            if (typeof modifierReplacer === 'function') {
                filter += `$${options.map(modifierReplacer).join(',')}`;
            }
            else {
                filter += `$${options.join(',')}`;
            }
        }
        return filter;
    }
    // Public API (Read-Only)
    getIdWithoutBadFilter() {
        // This method computes the id ignoring the $badfilter option (which will
        // correspond to the ID of filters being discarded). This allows us to
        // eliminate bad filters by comparing IDs, which is more robust and faster
        // than string comparison.
        return computeFilterId(this.mask & ~NETWORK_FILTER_MASK.isBadFilter, this.filter, this.hostname, this.domains, this.denyallow, this.optionValue);
    }
    getId() {
        if (this.id === undefined) {
            this.id = computeFilterId(this.mask, this.filter, this.hostname, this.domains, this.denyallow, this.optionValue);
        }
        return this.id;
    }
    hasFilter() {
        return this.filter !== undefined;
    }
    hasDomains() {
        return this.domains !== undefined;
    }
    getMask() {
        return this.mask;
    }
    getCptMask() {
        return this.getMask() & FROM_ANY;
    }
    isRedirect() {
        return getBit(this.getMask(), NETWORK_FILTER_MASK.isRedirect);
    }
    isRedirectRule() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isRedirectRule);
    }
    getRedirect() {
        var _a;
        return (_a = this.optionValue) !== null && _a !== void 0 ? _a : '';
    }
    isReplace() {
        return getBit(this.getMask(), NETWORK_FILTER_MASK.isReplace);
    }
    // Expected to be called only with `$replace` modifiers
    getHtmlModifier() {
        var _a;
        // Empty `$replace` modifier is to disable all replace modifiers on exception
        // This is checked on the parse time
        if (((_a = this.optionValue) === null || _a === void 0 ? void 0 : _a.length) === 0) {
            return null;
        }
        return replaceOptionValueToRegexp(this.optionValue);
    }
    isHtmlFilteringRule() {
        return this.isReplace();
    }
    getRedirectResource() {
        const redirect = this.getRedirect();
        const priorityIndex = redirect.lastIndexOf(':');
        if (priorityIndex === -1) {
            return redirect;
        }
        return redirect.slice(0, priorityIndex);
    }
    getRedirectPriority() {
        const redirect = this.getRedirect();
        const priorityIndex = redirect.lastIndexOf(':');
        if (priorityIndex === -1) {
            return 0;
        }
        return Number(redirect.slice(priorityIndex + 1));
    }
    hasHostname() {
        return this.hostname !== undefined;
    }
    getHostname() {
        return this.hostname || '';
    }
    getFilter() {
        return this.filter || '';
    }
    getRegex() {
        if (this.regex === undefined) {
            this.regex =
                this.filter !== undefined && this.isRegex()
                    ? compileRegex(this.filter, this.isLeftAnchor(), this.isRightAnchor(), this.isFullRegex())
                    : MATCH_ALL;
        }
        return this.regex;
    }
    getTokens() {
        TOKENS_BUFFER.reset();
        // If there is only one domain and no domain negation, we also use this
        // domain as a token.
        if (this.domains !== undefined &&
            this.domains.hostnames !== undefined &&
            this.domains.entities === undefined &&
            this.domains.notHostnames === undefined &&
            this.domains.notEntities === undefined &&
            this.domains.hostnames.length === 1) {
            TOKENS_BUFFER.push(this.domains.hostnames[0]);
        }
        // Get tokens from filter
        if (this.isFullRegex() === false) {
            if (this.filter !== undefined) {
                const skipLastToken = !this.isRightAnchor();
                const skipFirstToken = !this.isLeftAnchor();
                tokenizeWithWildcardsInPlace(this.filter, skipFirstToken, skipLastToken, TOKENS_BUFFER);
            }
            // Append tokens from hostname, if any
            if (this.hostname !== undefined) {
                tokenizeInPlace(this.hostname, false, this.filter !== undefined && this.filter.charCodeAt(0) === 42 /* '*' */, TOKENS_BUFFER);
            }
        }
        else if (this.filter !== undefined) {
            tokenizeRegexInPlace(this.filter, TOKENS_BUFFER);
        }
        // If we got no tokens for the filter/hostname part, then we will dispatch
        // this filter in multiple buckets based on the domains option.
        if (TOKENS_BUFFER.empty() === true &&
            this.domains !== undefined &&
            this.domains.hostnames !== undefined &&
            this.domains.entities === undefined &&
            this.domains.notHostnames === undefined &&
            this.domains.notEntities === undefined) {
            const result = [];
            for (const hostname of this.domains.hostnames) {
                const arr = new Uint32Array(1);
                arr[0] = hostname;
                result.push(arr);
            }
            return result;
        }
        // Add optional token for types
        if (TOKENS_BUFFER.empty() === true && this.fromAny() === false) {
            const types = getListOfRequestTypes(this);
            if (types.length !== 0) {
                const result = [];
                for (const type of types) {
                    const arr = new Uint32Array(1);
                    arr[0] = NORMALIZED_TYPE_TOKEN[type];
                    result.push(arr);
                }
                return result;
            }
        }
        // Add optional token for protocol
        if (this.fromHttp() === true && this.fromHttps() === false) {
            TOKENS_BUFFER.push(HTTP_HASH);
        }
        else if (this.fromHttps() === true && this.fromHttp() === false) {
            TOKENS_BUFFER.push(HTTPS_HASH);
        }
        return [TOKENS_BUFFER.slice()];
    }
    /**
     * Check if this filter should apply to a request with this content type.
     */
    isCptAllowed(cpt) {
        const mask = REQUEST_TYPE_TO_MASK[cpt];
        if (mask !== undefined) {
            return getBit(this.mask, mask);
        }
        // If content type is not supported (or not specified), we return `true`
        // only if the filter does not specify any resource type.
        return this.fromAny();
    }
    isException() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isException);
    }
    isHostnameAnchor() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isHostnameAnchor);
    }
    isRightAnchor() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isRightAnchor);
    }
    isLeftAnchor() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isLeftAnchor);
    }
    isImportant() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isImportant);
    }
    isFullRegex() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isFullRegex);
    }
    isRegex() {
        return (getBit(this.mask, NETWORK_FILTER_MASK.isRegex) ||
            getBit(this.mask, NETWORK_FILTER_MASK.isFullRegex));
    }
    isPlain() {
        return !this.isRegex();
    }
    isCSP() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isCSP);
    }
    isElemHide() {
        return this.isSpecificHide() && this.isGenericHide();
    }
    isSpecificHide() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isSpecificHide);
    }
    isGenericHide() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isGenericHide);
    }
    isBadFilter() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isBadFilter);
    }
    isUnicode() {
        return getBit(this.mask, NETWORK_FILTER_MASK.isUnicode);
    }
    fromAny() {
        return this.getCptMask() === FROM_ANY;
    }
    thirdParty() {
        return getBit(this.mask, NETWORK_FILTER_MASK.thirdParty);
    }
    firstParty() {
        return getBit(this.mask, NETWORK_FILTER_MASK.firstParty);
    }
    fromImage() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromImage);
    }
    fromMedia() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromMedia);
    }
    fromObject() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromObject);
    }
    fromOther() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromOther);
    }
    fromPing() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromPing);
    }
    fromScript() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromScript);
    }
    fromStylesheet() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromStylesheet);
    }
    fromDocument() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromDocument);
    }
    fromSubdocument() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromSubdocument);
    }
    fromWebsocket() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromWebsocket);
    }
    fromHttp() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromHttp);
    }
    fromHttps() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromHttps);
    }
    fromXmlHttpRequest() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromXmlHttpRequest);
    }
    fromFont() {
        return getBit(this.mask, NETWORK_FILTER_MASK.fromFont);
    }
}
// ---------------------------------------------------------------------------
// Filter parsing
// ---------------------------------------------------------------------------
function setNetworkMask(mask, m, value) {
    if (value === true) {
        return setBit(mask, m);
    }
    return clearBit(mask, m);
}
/**
 * Check if the sub-string contained between the indices start and end is a
 * regex filter (it contains a '*' or '^' char).
 */
function checkIsRegex(filter, start, end) {
    const indexOfSeparator = filter.indexOf('^', start);
    if (indexOfSeparator !== -1 && indexOfSeparator < end) {
        return true;
    }
    const indexOfWildcard = filter.indexOf('*', start);
    return indexOfWildcard !== -1 && indexOfWildcard < end;
}
/**
 * Handle hostname anchored filters, given 'hostname' from ||hostname and
 * request's hostname, check if there is a match. This is tricky because
 * filters authors rely and different assumptions. We can have prefix of suffix
 * matches of anchor.
 */
function isAnchoredByHostname(filterHostname, hostname, isFollowedByWildcard) {
    // Corner-case, if `filterHostname` is empty, then it's a match
    if (filterHostname.length === 0) {
        return true;
    }
    // `filterHostname` cannot be longer than actual hostname
    if (filterHostname.length > hostname.length) {
        return false;
    }
    // If they have the same length, they should be equal
    if (filterHostname.length === hostname.length) {
        return filterHostname === hostname;
    }
    // Check if `filterHostname` appears anywhere in `hostname`
    const matchIndex = hostname.indexOf(filterHostname);
    // No match
    if (matchIndex === -1) {
        return false;
    }
    // `filterHostname` is a prefix of `hostname` and needs to match full a label.
    //
    // Examples (filterHostname, hostname):
    //   * (foo, foo.com)
    //   * (sub.foo, sub.foo.com)
    if (matchIndex === 0) {
        return (isFollowedByWildcard === true ||
            hostname.charCodeAt(filterHostname.length) === 46 /* '.' */ ||
            filterHostname.charCodeAt(filterHostname.length - 1) === 46 /* '.' */);
    }
    // `filterHostname` is a suffix of `hostname`.
    //
    // Examples (filterHostname, hostname):
    //    * (foo.com, sub.foo.com)
    //    * (com, foo.com)
    if (hostname.length === matchIndex + filterHostname.length) {
        return (hostname.charCodeAt(matchIndex - 1) === 46 /* '.' */ ||
            filterHostname.charCodeAt(0) === 46 /* '.' */);
    }
    // `filterHostname` is infix of `hostname` and needs match full labels
    return ((isFollowedByWildcard === true ||
        hostname.charCodeAt(filterHostname.length) === 46 /* '.' */ ||
        filterHostname.charCodeAt(filterHostname.length - 1) === 46) /* '.' */ &&
        (hostname.charCodeAt(matchIndex - 1) === 46 || filterHostname.charCodeAt(0) === 46));
}
/**
 * Specialize a network filter depending on its type. It allows for more
 * efficient matching function.
 */
function checkPattern(filter, request) {
    const pattern = filter.getFilter();
    if (filter.isHostnameAnchor() === true) {
        // Make sure request is anchored by hostname before proceeding to matching
        const filterHostname = filter.getHostname();
        if (isAnchoredByHostname(filterHostname, request.hostname, filter.filter !== undefined && filter.filter.charCodeAt(0) === 42 /* '*' */) === false) {
            return false;
        }
        // At this point we know request is hostname anchored so we match the rest of the filter.
        if (filter.isRegex()) {
            // ||pattern*^
            return filter
                .getRegex()
                .test(request.url.slice(request.url.indexOf(filterHostname) + filterHostname.length));
        }
        else if (filter.isRightAnchor() && filter.isLeftAnchor()) {
            // |||pattern|
            // Since this is not a regex, the filter pattern must follow the hostname
            // with nothing in between. So we extract the part of the URL following
            // after hostname and will perform the matching on it.
            const urlAfterHostname = request.url.slice(request.url.indexOf(filterHostname) + filterHostname.length);
            // Since it must follow immediatly after the hostname and be a suffix of
            // the URL, we conclude that filter must be equal to the part of the
            // url following the hostname.
            return pattern === urlAfterHostname;
        }
        else if (filter.isRightAnchor()) {
            // ||pattern|
            const requestHostname = request.hostname;
            if (filter.hasFilter() === false) {
                // In this specific case it means that the specified hostname should match
                // at the end of the hostname of the request. This allows to prevent false
                // positive like ||foo.bar which would match https://foo.bar.baz where
                // ||foo.bar^ would not.
                return (filterHostname.length === requestHostname.length ||
                    requestHostname.endsWith(filterHostname));
            }
            else {
                // pattern|
                return request.url.endsWith(pattern);
            }
        }
        else if (filter.isLeftAnchor()) {
            // ||pattern + left-anchor => This means that a plain pattern needs to appear
            // exactly after the hostname, with nothing in between.
            // Since this is not a regex, the filter pattern must follow the hostname
            // with nothing in between. So we extract the part of the URL following
            // after hostname and will perform the matching on it.
            return fastStartsWithFrom(request.url, pattern, request.url.indexOf(filterHostname) + filterHostname.length);
        }
        if (filter.hasFilter() === false) {
            return true;
        }
        // We consider this a match if the plain patter (i.e.: filter) appears anywhere.
        return (request.url.indexOf(pattern, request.url.indexOf(filterHostname) + filterHostname.length) !==
            -1);
    }
    else if (filter.isRegex()) {
        // pattern*^
        return filter.getRegex().test(request.url);
    }
    else if (filter.isLeftAnchor() && filter.isRightAnchor()) {
        // |pattern|
        return request.url === pattern;
    }
    else if (filter.isLeftAnchor()) {
        // |pattern
        return fastStartsWith(request.url, pattern);
    }
    else if (filter.isRightAnchor()) {
        // pattern|
        return request.url.endsWith(pattern);
    }
    // pattern
    if (filter.hasFilter() === false) {
        return true;
    }
    return request.url.indexOf(pattern) !== -1;
}
function checkOptions(filter, request) {
    // We first discard requests based on type, protocol and party. This is really
    // cheap and should be done first.
    if (filter.isCptAllowed(request.type) === false ||
        (request.isHttps === true && filter.fromHttps() === false) ||
        (request.isHttp === true && filter.fromHttp() === false) ||
        (filter.firstParty() === false && request.isFirstParty === true) ||
        (filter.thirdParty() === false && request.isThirdParty === true)) {
        return false;
    }
    // If `sourceHostname` is *not* matched by `domain` then the request should be allowed.
    if (filter.domains !== undefined &&
        filter.domains.match(request.sourceHostnameHashes, request.sourceEntityHashes) === false) {
        return false;
    }
    // If `hostname` is matched by `denyallow` then the request should be allowed.
    if (filter.denyallow !== undefined &&
        filter.denyallow.match(request.getHostnameHashes(), request.getEntityHashes()) === true) {
        return false;
    }
    return true;
}

export { NETWORK_FILTER_MASK, NetworkFilter as default, isAnchoredByHostname, replaceOptionValueToRegexp };
