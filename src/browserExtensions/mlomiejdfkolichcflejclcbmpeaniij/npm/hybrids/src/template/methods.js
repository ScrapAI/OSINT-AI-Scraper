function key(id) {
  this.id = id;
  return this;
}

function style(...styles) {
  this.styleSheets = this.styleSheets || [];
  this.styleSheets.push(...styles);

  return this;
}

function css(parts, ...args) {
  this.styleSheets = this.styleSheets || [];

  let result = parts[0];
  for (let index = 1; index < parts.length; index++) {
    result +=
      (args[index - 1] !== undefined ? args[index - 1] : "") + parts[index];
  }

  this.styleSheets.push(result);

  return this;
}

function use(plugin) {
  this.plugins = this.plugins || [];
  this.plugins.push(plugin);

  return this;
}

export { css, key, style, use };
