/*!
 * Copyright (c) 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
function extractHTMLSelectorFromRule(rule) {
    if (rule.startsWith('^script') === false) {
        return undefined;
    }
    const prefix = ':has-text(';
    const selectors = [];
    let index = 7;
    // ^script:has-text
    //        ^ 7
    // Prepare for finding one or more ':has-text(' selectors in a row
    while (rule.startsWith(prefix, index)) {
        index += prefix.length;
        let currentParsingDepth = 1;
        const startOfSelectorIndex = index;
        let prev = -1; // previous character
        for (; index < rule.length && currentParsingDepth !== 0; index += 1) {
            const code = rule.charCodeAt(index);
            if (prev !== 92 /* '\' */) {
                if (code === 40 /* '(' */) {
                    currentParsingDepth += 1;
                }
                if (code === 41 /* ')' */) {
                    currentParsingDepth -= 1;
                }
            }
            prev = code;
        }
        selectors.push(rule.slice(startOfSelectorIndex, index - 1));
    }
    if (index !== rule.length) {
        return undefined;
    }
    return ['script', selectors];
}

export { extractHTMLSelectorFromRule };
