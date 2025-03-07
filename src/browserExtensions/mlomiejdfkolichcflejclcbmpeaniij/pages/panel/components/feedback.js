import { html } from '../../../npm/hybrids/src/template/index.js';

/**
 * Ghostery Browser Extension
 * https://www.ghostery.com/
 *
 * Copyright 2017-present Ghostery GmbH. All rights reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0
 */


const __vite_glob_0_5 = {
  blocked: 0,
  modified: 0,
  render: ({ blocked, modified }) => html`
    <template layout="row padding:1:0:1">
      ${blocked > 0 &&
      html`
        <section layout="column center grow">
          <div layout="row center gap:0.5">
            <ui-icon name="block-s" color="danger-700"></ui-icon>
            <ui-text type="headline-s">${blocked}</ui-text>
          </div>
          <div layout="row center gap:0.5">
            <ui-text type="label-xs">Trackers blocked</ui-text>
          </div>
        </section>
      `}
      ${modified > 0 &&
      html`
        <section layout="column center grow">
          <div layout="row center gap:0.5">
            <ui-icon name="eye" color="primary-700"></ui-icon>
            <ui-text type="headline-s">${modified}</ui-text>
          </div>
          <div layout="row center gap:0.5">
            <ui-text type="label-xs">Trackers modified</ui-text>
          </div>
        </section>
      `}
    </template>
  `.css`
    :host {
      background: var(--ui-color-gray-100);
    }
  `,
};

export { __vite_glob_0_5 as default };
