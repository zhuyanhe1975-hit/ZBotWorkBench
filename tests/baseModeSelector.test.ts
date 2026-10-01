import assert from 'node:assert/strict';
import test from 'node:test';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ConfigurationEditor } from '../src/components/ConfigurationEditor';
import { PRESET_CONFIGURATIONS } from '../src/data/presets';
import { SEVEN_DOF_PRESETS } from '../src/data/sevenDofPresets';
import { createConfiguration, createCubeConfiguration, createCubeQuadruped, createTetrahedronConfiguration } from '../src/utils/configuration';

const configurations = [...PRESET_CONFIGURATIONS, ...SEVEN_DOF_PRESETS,
  createConfiguration(), createCubeConfiguration(), createCubeQuadruped(), createTetrahedronConfiguration()];

for (const config of configurations) {
  test(`${config.name} exposes both base modes without changing tabs`, () => {
    for (const baseMode of ['fixed', 'free'] as const) {
      const html = renderToStaticMarkup(React.createElement(ConfigurationEditor, {
        currentConfig: { ...config, baseMode }, onConfigChange: () => {},
        onOpenXmlModal: () => {}, onOpenMeshModal: () => {},
      }));
      const selector = html.match(/<select aria-label="基座模式"[\s\S]*?<\/select>/)?.[0];
      assert.ok(selector);
      assert.match(selector, /value="fixed"/);
      assert.match(selector, /value="free"/);
      assert.match(selector, new RegExp(`value="${baseMode}" selected=""`));
    }
  });
}
