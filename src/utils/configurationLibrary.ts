import { ZbotConfiguration } from '../types/zbot';
import { parseConfiguration } from './configuration';

export const CONFIGURATION_LIBRARY_KEY = 'zbot.configurations.v1';
const LEGACY_KEY = 'zbot.configuration.v1';
type LibraryStorage = Pick<Storage, 'getItem' | 'setItem'>;

export function readConfigurationLibrary(storage: LibraryStorage): ZbotConfiguration[] {
  const text = storage.getItem(CONFIGURATION_LIBRARY_KEY);
  if (text !== null) {
    const entries: unknown = JSON.parse(text);
    if (!Array.isArray(entries)) throw new Error('已保存构型列表格式错误');
    return entries.map(entry => parseConfiguration(JSON.stringify(entry)));
  }
  const legacy = storage.getItem(LEGACY_KEY);
  if (!legacy) return [];
  const entries = [parseConfiguration(legacy)];
  storage.setItem(CONFIGURATION_LIBRARY_KEY, JSON.stringify(entries));
  return entries;
}

export function saveConfigurationToLibrary(storage: LibraryStorage, config: ZbotConfiguration): ZbotConfiguration[] {
  const saved = parseConfiguration(JSON.stringify({ ...config, id: `saved-${typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : Array.from(crypto.getRandomValues(new Uint32Array(4)), value => value.toString(16).padStart(8, '0')).join('')}`, category: 'custom' }));
  const entries = [saved, ...readConfigurationLibrary(storage)];
  storage.setItem(CONFIGURATION_LIBRARY_KEY, JSON.stringify(entries));
  return entries;
}

export function deleteConfigurationFromLibrary(storage: LibraryStorage, id: string): ZbotConfiguration[] {
  const entries = readConfigurationLibrary(storage).filter(config => config.id !== id);
  storage.setItem(CONFIGURATION_LIBRARY_KEY, JSON.stringify(entries));
  return entries;
}
