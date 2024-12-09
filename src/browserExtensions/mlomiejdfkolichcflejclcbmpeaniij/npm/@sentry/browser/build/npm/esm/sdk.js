import { BrowserClient } from './client.js';
import { DEBUG_BUILD } from './debug-build.js';
import { WINDOW } from './helpers.js';
import { breadcrumbsIntegration } from './integrations/breadcrumbs.js';
import { browserApiErrorsIntegration } from './integrations/browserapierrors.js';
import { globalHandlersIntegration } from './integrations/globalhandlers.js';
import { httpContextIntegration } from './integrations/httpcontext.js';
import { linkedErrorsIntegration } from './integrations/linkederrors.js';
import { defaultStackParser } from './stack-parsers.js';
import { makeFetchTransport } from './transports/fetch.js';
import { consoleSandbox, logger } from '../../../../utils/build/esm/logger.js';
import { supportsFetch } from '../../../../utils/build/esm/supports.js';
import { stackParserFromStackParserOptions } from '../../../../utils/build/esm/stacktrace.js';
import { getIntegrationsToSetup } from '../../../../core/build/esm/integration.js';
import { initAndBind } from '../../../../core/build/esm/sdk.js';
import { startSession, captureSession } from '../../../../core/build/esm/exports.js';
import { addHistoryInstrumentationHandler } from '../../../../../@sentry-internal/browser-utils/build/esm/instrument/history.js';
import { inboundFiltersIntegration } from '../../../../core/build/esm/integrations/inboundfilters.js';
import { functionToStringIntegration } from '../../../../core/build/esm/integrations/functiontostring.js';
import { dedupeIntegration } from '../../../../core/build/esm/integrations/dedupe.js';

/** Get the default integrations for the browser SDK. */
function getDefaultIntegrations(_options) {
  /**
   * Note: Please make sure this stays in sync with Angular SDK, which re-exports
   * `getDefaultIntegrations` but with an adjusted set of integrations.
   */
  return [
    inboundFiltersIntegration(),
    functionToStringIntegration(),
    browserApiErrorsIntegration(),
    breadcrumbsIntegration(),
    globalHandlersIntegration(),
    linkedErrorsIntegration(),
    dedupeIntegration(),
    httpContextIntegration(),
  ];
}

function applyDefaultOptions(optionsArg = {}) {
  const defaultOptions = {
    defaultIntegrations: getDefaultIntegrations(),
    release:
      typeof __SENTRY_RELEASE__ === 'string' // This allows build tooling to find-and-replace __SENTRY_RELEASE__ to inject a release value
        ? __SENTRY_RELEASE__
        : WINDOW.SENTRY_RELEASE && WINDOW.SENTRY_RELEASE.id // This supports the variable that sentry-webpack-plugin injects
          ? WINDOW.SENTRY_RELEASE.id
          : undefined,
    autoSessionTracking: true,
    sendClientReports: true,
  };

  // TODO: Instead of dropping just `defaultIntegrations`, we should simply
  // call `dropUndefinedKeys` on the entire `optionsArg`.
  // However, for this to work we need to adjust the `hasTracingEnabled()` logic
  // first as it differentiates between `undefined` and the key not being in the object.
  if (optionsArg.defaultIntegrations == null) {
    delete optionsArg.defaultIntegrations;
  }

  return { ...defaultOptions, ...optionsArg };
}

function shouldShowBrowserExtensionError() {
  const windowWithMaybeExtension =
    typeof WINDOW.window !== 'undefined' && (WINDOW );
  if (!windowWithMaybeExtension) {
    // No need to show the error if we're not in a browser window environment (e.g. service workers)
    return false;
  }

  const extensionKey = windowWithMaybeExtension.chrome ? 'chrome' : 'browser';
  const extensionObject = windowWithMaybeExtension[extensionKey];

  const runtimeId = extensionObject && extensionObject.runtime && extensionObject.runtime.id;
  const href = (WINDOW.location && WINDOW.location.href) || '';

  const extensionProtocols = ['chrome-extension:', 'moz-extension:', 'ms-browser-extension:', 'safari-web-extension:'];

  // Running the SDK in a dedicated extension page and calling Sentry.init is fine; no risk of data leakage
  const isDedicatedExtensionPage =
    !!runtimeId && WINDOW === WINDOW.top && extensionProtocols.some(protocol => href.startsWith(`${protocol}//`));

  // Running the SDK in NW.js, which appears like a browser extension but isn't, is also fine
  // see: https://github.com/getsentry/sentry-javascript/issues/12668
  const isNWjs = typeof windowWithMaybeExtension.nw !== 'undefined';

  return !!runtimeId && !isDedicatedExtensionPage && !isNWjs;
}

/**
 * A magic string that build tooling can leverage in order to inject a release value into the SDK.
 */

/**
 * The Sentry Browser SDK Client.
 *
 * To use this SDK, call the {@link init} function as early as possible when
 * loading the web page. To set context information or send manual events, use
 * the provided methods.
 *
 * @example
 *
 * ```
 *
 * import { init } from '@sentry/browser';
 *
 * init({
 *   dsn: '__DSN__',
 *   // ...
 * });
 * ```
 *
 * @example
 * ```
 *
 * import { addBreadcrumb } from '@sentry/browser';
 * addBreadcrumb({
 *   message: 'My Breadcrumb',
 *   // ...
 * });
 * ```
 *
 * @example
 *
 * ```
 *
 * import * as Sentry from '@sentry/browser';
 * Sentry.captureMessage('Hello, world!');
 * Sentry.captureException(new Error('Good bye'));
 * Sentry.captureEvent({
 *   message: 'Manual',
 *   stacktrace: [
 *     // ...
 *   ],
 * });
 * ```
 *
 * @see {@link BrowserOptions} for documentation on configuration options.
 */
function init(browserOptions = {}) {
  const options = applyDefaultOptions(browserOptions);

  if (!options.skipBrowserExtensionCheck && shouldShowBrowserExtensionError()) {
    consoleSandbox(() => {
      // eslint-disable-next-line no-console
      console.error(
        '[Sentry] You cannot run Sentry this way in a browser extension, check: https://docs.sentry.io/platforms/javascript/best-practices/browser-extensions/',
      );
    });
    return;
  }

  if (DEBUG_BUILD) {
    if (!supportsFetch()) {
      logger.warn(
        'No Fetch API detected. The Sentry SDK requires a Fetch API compatible environment to send events. Please add a Fetch API polyfill.',
      );
    }
  }
  const clientOptions = {
    ...options,
    stackParser: stackParserFromStackParserOptions(options.stackParser || defaultStackParser),
    integrations: getIntegrationsToSetup(options),
    transport: options.transport || makeFetchTransport,
  };

  const client = initAndBind(BrowserClient, clientOptions);

  if (options.autoSessionTracking) {
    startSessionTracking();
  }

  return client;
}

/**
 * Enable automatic Session Tracking for the initial page load.
 */
function startSessionTracking() {
  if (typeof WINDOW.document === 'undefined') {
    DEBUG_BUILD && logger.warn('Session tracking in non-browser environment with @sentry/browser is not supported.');
    return;
  }

  // The session duration for browser sessions does not track a meaningful
  // concept that can be used as a metric.
  // Automatically captured sessions are akin to page views, and thus we
  // discard their duration.
  startSession({ ignoreDuration: true });
  captureSession();

  // We want to create a session for every navigation as well
  addHistoryInstrumentationHandler(({ from, to }) => {
    // Don't create an additional session for the initial route or if the location did not change
    if (from !== undefined && from !== to) {
      startSession({ ignoreDuration: true });
      captureSession();
    }
  });
}

export { getDefaultIntegrations, init };