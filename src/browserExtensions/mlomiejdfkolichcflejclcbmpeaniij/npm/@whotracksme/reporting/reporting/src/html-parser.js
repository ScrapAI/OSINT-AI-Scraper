import '../../../../linkedom/esm/cached.js';
import { DOMParser } from '../../../../linkedom/esm/dom/parser.js';
import '../../../../linkedom/esm/index.js';

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


function parseHtml(html) {
  const domParser = new DOMParser();
  return domParser.parseFromString(html, 'text/html');
}

export { parseHtml as default };
