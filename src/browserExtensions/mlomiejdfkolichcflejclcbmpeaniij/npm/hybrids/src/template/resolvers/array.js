import { removeTemplate, getMeta, getTemplateEnd } from '../utils.js';

const arrayMap = new WeakMap();

function movePlaceholder(target, previousSibling) {
  const meta = getMeta(target);
  const startNode = meta.startNode;
  const endNode = getTemplateEnd(meta.endNode);

  previousSibling.parentNode.insertBefore(target, previousSibling.nextSibling);

  let prevNode = target;
  let node = startNode;
  while (node) {
    const nextNode = node.nextSibling;
    prevNode.parentNode.insertBefore(node, prevNode.nextSibling);
    prevNode = node;
    node = nextNode !== endNode.nextSibling && nextNode;
  }
}

function resolveArray(
  host,
  target,
  value,
  resolveValue,
  useLayout,
) {
  let lastEntries = arrayMap.get(target);
  const entries = value.map((item, index) => ({
    id: hasOwnProperty.call(item, "id") ? item.id : index,
    value: item,
    placeholder: null,
    available: true,
  }));

  arrayMap.set(target, entries);

  if (lastEntries) {
    const ids = new Set();
    for (const entry of entries) {
      ids.add(entry.id);
    }

    lastEntries = lastEntries.filter((entry) => {
      if (!ids.has(entry.id)) {
        removeTemplate(entry.placeholder);
        entry.placeholder.parentNode.removeChild(entry.placeholder);
        return false;
      }

      return true;
    });
  }

  let previousSibling = target;
  const lastIndex = value.length - 1;
  const meta = getMeta(target);

  for (let index = 0; index < entries.length; index += 1) {
    const entry = entries[index];
    let matchedEntry;
    if (lastEntries) {
      for (let i = 0; i < lastEntries.length; i += 1) {
        if (lastEntries[i].available && lastEntries[i].id === entry.id) {
          matchedEntry = lastEntries[i];
          break;
        }
      }
    }

    if (matchedEntry) {
      matchedEntry.available = false;
      entry.placeholder = matchedEntry.placeholder;

      if (entry.placeholder.previousSibling !== previousSibling) {
        movePlaceholder(entry.placeholder, previousSibling);
      }
      if (matchedEntry.value !== entry.value) {
        resolveValue(
          host,
          entry.placeholder,
          entry.value,
          matchedEntry.value,
          useLayout,
        );
      }
    } else {
      entry.placeholder = globalThis.document.createTextNode("");
      previousSibling.parentNode.insertBefore(
        entry.placeholder,
        previousSibling.nextSibling,
      );
      resolveValue(host, entry.placeholder, entry.value, undefined, useLayout);
    }

    previousSibling = getTemplateEnd(
      getMeta(entry.placeholder).endNode || entry.placeholder,
    );

    if (index === 0) meta.startNode = entry.placeholder;
    if (index === lastIndex) meta.endNode = previousSibling;
  }

  if (lastEntries) {
    for (const entry of lastEntries) {
      if (entry.available) {
        removeTemplate(entry.placeholder);
        entry.placeholder.parentNode.removeChild(entry.placeholder);
      }
    }
  }
}

export { arrayMap, resolveArray as default };
