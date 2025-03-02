import '../../../adblocker/dist/esm/data-view.js';
export { adsAndTrackingLists, adsLists, fetchLists, fetchResources, fetchWithRetry, fullLists } from '../../../adblocker/dist/esm/fetch.js';
export { default as CosmeticFilter } from '../../../adblocker/dist/esm/filters/cosmetic.js';
export { FilterType, detectFilterType, getLinesWithFilters, mergeDiffs, parseFilter, parseFilters } from '../../../adblocker/dist/esm/lists.js';
export { default as Request, getHostnameHashesFromLabelsBackward } from '../../../adblocker/dist/esm/request.js';
import '../../../../@remusao/small/dist/esm/index.js';
export { default as NetworkFilter } from '../../../adblocker/dist/esm/filters/network.js';
import '../../../adblocker/dist/esm/preprocessor.js';

/*!
 * Copyright (c) 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */
function isFirefox() {
    try {
        return navigator.userAgent.indexOf('Firefox') !== -1;
    }
    catch (e) {
        return false;
    }
}
/**
 * There are different ways to inject scriptlets ("push" vs "pull").
 * This function should decide based on the environment what to use:
 *
 * 1) "Pushing" means the adblocker will listen on "onCommitted" events
 *    and then execute scripts by running the tabs.executeScript API.
 * 2) "Pulling" means the adblocker will inject a content script, which
 *    runs before the page loads (and on the DOM changes), fetches
 *    scriplets from the background and runs them.
 *
 * Note:
 * - the "push" model requires permission to the webNavigation API.
 *   If that is not available, the implementation will fall back to the
 *   "pull" model, which does not have this requirement.
 */
function usePushScriptsInjection() {
    // There is no fundamental reason why it should not work on Firefox,
    // but given that there are no known issues with Firefox, let's keep
    // the old, proven technique until there is evidence that changes
    // are needed.
    //
    // Take YouTube as an example: on Chrome (or forks like Edge), the adblocker
    // will sometimes fail to block ads if you reload the page multiple times;
    // on Firefox, the same steps do not seem to trigger any ads.
    return !isFirefox();
}
usePushScriptsInjection();
