/**
 * Use this attribute to represent the source of a span.
 * Should be one of: custom, url, route, view, component, task, unknown
 *
 */
const SEMANTIC_ATTRIBUTE_SENTRY_SOURCE = 'sentry.source';

/**
 * Use this attribute to represent the sample rate used for a span.
 */
const SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE = 'sentry.sample_rate';

/**
 * Use this attribute to represent the operation of a span.
 */
const SEMANTIC_ATTRIBUTE_SENTRY_OP = 'sentry.op';

/**
 * Use this attribute to represent the origin of a span.
 */
const SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN = 'sentry.origin';

export { SEMANTIC_ATTRIBUTE_SENTRY_OP, SEMANTIC_ATTRIBUTE_SENTRY_ORIGIN, SEMANTIC_ATTRIBUTE_SENTRY_SAMPLE_RATE, SEMANTIC_ATTRIBUTE_SENTRY_SOURCE };
