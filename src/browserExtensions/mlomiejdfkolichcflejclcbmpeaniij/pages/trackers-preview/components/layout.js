import { sortCategories } from '../../../ui/categories.js';
import DisablePreviewImg from '../assets/disable-preview.svg.js';
import Options from '../../../store/options.js';
import store from '../../../npm/hybrids/src/store.js';
import { html } from '../../../npm/hybrids/src/template/index.js';
import { dispatch } from '../../../npm/hybrids/src/utils.js';

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


const sort = sortCategories();

const __vite_glob_0_0 = {
  confirmDisabled: false,
  stats: undefined,
  domain: '',
  options: store(Options),
  render: ({ domain, confirmDisabled, stats, options }) => html`
    <template layout="block height:full">
      ${confirmDisabled
        ? html`
            <main layout="column gap:2 padding:3:5:3">
              <img
                src="${DisablePreviewImg}"
                alt="Disable Preview Trackers"
                layout="self:center"
              />
              <div layout="block:center column gap">
                <ui-text type="label-l">
                  Are you sure you want to disable Trackers Preview?
                </ui-text>
                <ui-text>
                  You will no longer see tracker wheels next to the search
                  results.
                </ui-text>
              </div>
              <div layout="grid:2 gap:2">
                <ui-button>
                  <button onclick="${html.set('confirmDisabled', false)}">
                    Cancel
                  </button>
                </ui-button>
                <ui-button id="disable">
                  <button onclick="${(host) => dispatch(host, 'disable')}">
                    Disable
                  </button>
                </ui-button>
              </div>
            </main>
          `
        : html`
            <ui-header>
              <ui-icon name="logo" slot="icon" layout="size:3"></ui-icon>
              <ui-text type="label-m">${domain}</ui-text>
              <ui-action slot="actions">
                <button
                  onclick="${(host) => dispatch(host, 'close')}"
                  layout="row center size:3"
                >
                  <ui-icon
                    name="close"
                    color="gray-800"
                    layout="size:2.5"
                  ></ui-icon>
                </button>
              </ui-action>
            </ui-header>

            <main layout="padding:1.5">
              ${stats &&
              html`
                <ui-stats
                  domain="${domain}"
                  categories="${stats.sort(sort)}"
                  layout="relative layer:101"
                  wtm-link
                >
                </ui-stats>
              `}
            </main>
            ${store.ready(options) &&
            !options.managed &&
            html`<footer layout="row center padding:2">
              <ui-action>
                <button
                  onclick="${html.set('confirmDisabled', true)}"
                  layout="row gap:0.5"
                >
                  <ui-icon name="block-s" color="gray-600"></ui-icon>
                  <ui-text type="label-s" color="gray-600">
                    Disable Trackers Preview
                  </ui-text>
                </button>
              </ui-action>
            </footer>`}
          `}
    </template>
  `.css`
    :host {
      border: 1px solid var(--ui-color-gray-300);
      border-radius: 16px;
      overflow: hidden;
    }

    footer {
      background: var(--ui-color-gray-100);
    }

    ui-button {
      text-transform: none;
      border-radius: 8px;
      box-shadow: 0px 2px 6px rgba(32, 44, 68, 0.08);
      --ui-button-color-hover: var(--ui-color-primary-700);
    }

    ui-button#disable {
      color: var(--ui-color-danger-500);
      --ui-button-color-hover: var(--ui-color-danger-700);
    }
  `,
};

export { __vite_glob_0_0 as default };
