import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createConfiguration } from '../src/utils/configuration';
import { CONFIGURATION_LIBRARY_KEY, readConfigurationLibrary, saveConfigurationToLibrary, deleteConfigurationFromLibrary } from '../src/utils/configurationLibrary';

function storage() {
  const data = new Map<string, string>();
  return { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
}

test('saving multiple configurations including equal names preserves independent snapshots after reload', () => {
  const store = storage(), config = createConfiguration();
  const first = saveConfigurationToLibrary(store, config)[0];
  config.modules[0].initialAngle = 30;
  const entries = saveConfigurationToLibrary(store, config);
  assert.equal(entries.length, 2);
  assert.notEqual(entries[0].id, first.id);
  assert.equal(entries[0].name, first.name);
  assert.equal(entries[0].modules[0].initialAngle, 30);
  assert.notEqual(entries[1].modules[0].initialAngle, 30);
  assert.deepEqual(readConfigurationLibrary(store), entries);
  assert.deepEqual(deleteConfigurationFromLibrary(store, first.id), [entries[0]]);
  assert.deepEqual(readConfigurationLibrary(store), [entries[0]]);
});

test('legacy single saved configuration migrates once and stays deleted', () => {
  const store = storage(), config = createConfiguration();
  store.setItem('zbot.configuration.v1', JSON.stringify(config));
  assert.deepEqual(readConfigurationLibrary(store), [config]);
  assert.ok(store.getItem(CONFIGURATION_LIBRARY_KEY));
  assert.equal(saveConfigurationToLibrary(store, config).length, 2);
  deleteConfigurationFromLibrary(store, config.id);
  const remaining = readConfigurationLibrary(store);
  deleteConfigurationFromLibrary(store, remaining[0].id);
  assert.deepEqual(readConfigurationLibrary(store), []);
});

test('invalid library or failed write cannot silently overwrite saved configurations', () => {
  const store = storage();
  store.setItem(CONFIGURATION_LIBRARY_KEY, '{}');
  assert.throws(() => saveConfigurationToLibrary(store, createConfiguration()), /格式错误/);
  assert.equal(store.getItem(CONFIGURATION_LIBRARY_KEY), '{}');
  const blocked = { getItem: () => null, setItem: () => { throw new Error('quota'); } };
  assert.throws(() => saveConfigurationToLibrary(blocked, createConfiguration()), /quota/);
});
